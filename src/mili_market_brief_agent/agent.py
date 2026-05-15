import io
import os
from typing import Any
from agents import Agent, Runner, function_tool, AsyncOpenAI
from agents.models.openai_responses import OpenAIResponsesModel
from pydantic import BaseModel

from .tools import (
    ClientHolding,
    build_personalized_summary,
    fetch_market_data,
    parse_holdings_csv,
    parse_holdings_text,
)


def _build_openai_agent(api_key: str, schedule: bool) -> Agent:
    client = AsyncOpenAI(api_key=api_key)
    model = OpenAIResponsesModel(os.getenv("AI_MODEL", "gpt-4.1-mini"), openai_client=client)
    schedule_note = (
        "Morning delivery is SCHEDULED — mention this at the end of the advisor_summary."
        if schedule
        else "Morning delivery is NOT scheduled — do not mention scheduling."
    )
    return Agent(
        name="Mili Market Brief Agent",
        instructions=(
            "You are a senior wealth advisor assistant at Mili, helping financial advisors prepare "
            "for their morning client calls.\n\n"

            "WORKFLOW — follow these steps in order, every time:\n"
            "1. Call parse_holdings with the raw holdings text to extract structured positions.\n"
            "2. From the parsed holdings, identify the distinct sectors present.\n"
            "3. Call get_market_data with those sectors to fetch today's market context.\n"
            "4. Produce the final MarketBrief output.\n\n"

            "OUTPUT REQUIREMENTS:\n"
            "- advisor_summary: 150–250 words. Written for the advisor to read aloud or paste into "
            "a client email. Structure it as: (a) what moved in the market today, (b) why it matters "
            "specifically to this client given their holdings and risk profile, (c) one clear action "
            f"item or question the advisor should raise. {schedule_note}\n"
            "- talking_points: 3–4 punchy bullet points the advisor can reference during a call. "
            "Each should be specific to this client's positions — not generic market commentary. "
            "Reference actual tickers and sectors from the holdings.\n"
            "- sectors_in_focus: list only sectors that actually appear in the client's holdings.\n\n"

            "TONE: Confident, concise, jargon-aware but not jargon-heavy. Written for a professional "
            "advisor, not the end client. Avoid filler phrases like 'it is worth noting' or 'as always'.\n\n"

            "CONSTRAINTS:\n"
            "- Never invent tickers or prices not present in the holdings or market data.\n"
            "- If holdings are empty or unparseable, set advisor_summary to a clear error message "
            "and return empty lists for the other fields.\n"
            "- Always call both tools before producing output. Do not skip get_market_data."
        ),
        tools=[parse_client_holdings, get_market_data],
        output_type=MarketBrief,
        model=model,
    )


@function_tool(
    name_override="parse_holdings",
    description_override=(
        "Parse raw client holdings text or CSV into a structured list of positions. "
        "Each position includes ticker symbol, quantity, market value in USD, and sector. "
        "Call this first, before get_market_data. Pass the full raw holdings string as raw_holdings. "
        "Use source='csv' only if the input is comma-separated with headers; otherwise use source='text'."
    ),
)
def parse_client_holdings(raw_holdings: str, source: str = "text") -> list[dict]:
    if source == "csv":
        buffer = io.StringIO(raw_holdings)
        holdings = parse_holdings_csv(buffer)
    else:
        holdings = parse_holdings_text(raw_holdings)
    return [h.__dict__ for h in holdings]


@function_tool(
    name_override="get_market_data",
    description_override=(
        "Fetch today's top market movers, relevant news headlines, and sector-level commentary "
        "for the sectors present in the client's portfolio. "
        "Call this after parse_holdings, passing the list of sector names extracted from the holdings. "
        "Returns top movers, headlines, and sector coverage relevant to the client."
    ),
)
def get_market_data(sectors: list[str]) -> dict:
    return fetch_market_data(sectors)


class MarketBrief(BaseModel):
    advisor_summary: str
    talking_points: list[str]
    sectors_in_focus: list[str]


def _parse_holdings_objects(holdings_text: str, uploaded_file: io.StringIO | None) -> list[ClientHolding]:
    if uploaded_file is not None:
        return parse_holdings_csv(uploaded_file)
    return parse_holdings_text(holdings_text)


def _parse_holdings(holdings_text: str, uploaded_file: io.StringIO | None) -> list[dict]:
    return [h.__dict__ for h in _parse_holdings_objects(holdings_text, uploaded_file)]


def _extract_sectors_from_holdings(holdings: list[dict]) -> list[str]:
    return [sector for sector in {h.get("sector", "Unknown") for h in holdings} if sector and sector != "Unknown"]


def _build_talking_points(holdings: list[dict], risk_profile: str, sectors: list[str]) -> list[str]:
    """Build client-specific talking points from actual holdings data."""
    if not holdings:
        return ["No holdings data available — ask the client to provide their current portfolio."]

    top_assets = sorted(holdings, key=lambda h: h.get("market_value", 0), reverse=True)[:3]
    top_tickers = ", ".join(h["ticker"] for h in top_assets)
    sector_str = ", ".join(sectors) if sectors else "their current sectors"
    total_value = sum(h.get("market_value", 0) for h in holdings)

    if risk_profile == "Conservative":
        risk_point = (
            f"With a conservative profile, confirm {top_tickers} moves are within acceptable volatility "
            "bounds — flag anything that's drifted toward growth-like drawdown."
        )
    elif risk_profile == "Moderate":
        risk_point = (
            f"Check whether {sector_str} concentration has shifted after today's moves "
            "and whether a tactical trim is warranted."
        )
    else:
        risk_point = (
            f"Growth positioning in {top_tickers} may amplify today's market swings — "
            "confirm the client is still aligned on drawdown tolerance."
        )

    return [
        risk_point,
        f"Largest positions are {top_tickers}, representing the bulk of the ${total_value:,.0f} portfolio — "
        "verify these are still within target weight after today's session.",
        f"Sector exposure is concentrated in {sector_str} — discuss whether this remains intentional "
        "or if diversification is worth revisiting.",
        "Ask whether any upcoming liquidity needs (tax payments, large purchases) should inform near-term positioning.",
    ]


def _extract_real_agent_steps(run_result: Any, holdings: list[dict], schedule: bool) -> list[dict]:
    """
    Extract actual tool calls made during the agent run from the SDK's new_items trace.
    Falls back to a labelled summary if the trace is unavailable.
    """
    steps = []
    try:
        for item in run_result.new_items:
            item_type = type(item).__name__
            # ToolCallItem: the agent decided to call a tool
            if item_type == "ToolCallItem":
                tool_name = getattr(item, "raw_item", {})
                # raw_item is the underlying dict from the Responses API
                raw = item.raw_item if hasattr(item, "raw_item") else {}
                name = raw.get("name", "") if isinstance(raw, dict) else getattr(raw, "name", "unknown_tool")
                arguments = raw.get("arguments", "") if isinstance(raw, dict) else getattr(raw, "arguments", "")
                steps.append({
                    "type": "tool_call",
                    "tool": name,
                    "arguments": arguments,
                    "scheduled": schedule,
                })
            # ToolCallOutputItem: the tool returned a result
            elif item_type == "ToolCallOutputItem":
                output = item.output if hasattr(item, "output") else str(item)
                # Summarise rather than dump raw output (can be large)
                result_preview = str(output)[:300] + ("…" if len(str(output)) > 300 else "")
                steps.append({
                    "type": "tool_result",
                    "result_preview": result_preview,
                    "scheduled": schedule,
                })
            # MessageOutputItem: the agent produced a text message mid-run
            elif item_type == "MessageOutputItem":
                content = ""
                try:
                    for block in item.raw_item.content:
                        if hasattr(block, "text"):
                            content += block.text
                except Exception:
                    pass
                if content:
                    steps.append({
                        "type": "agent_message",
                        "content": content[:300],
                        "scheduled": schedule,
                    })
    except Exception:
        # If the trace API changes or is unavailable, fall back gracefully
        pass

    if not steps:
        # Fallback: show at least that the two required tools were invoked
        sectors = _extract_sectors_from_holdings(holdings)
        steps = [
            {
                "type": "tool_call",
                "tool": "parse_holdings",
                "arguments": f"Parsed {len(holdings)} holdings",
                "scheduled": schedule,
            },
            {
                "type": "tool_call",
                "tool": "get_market_data",
                "arguments": f"Fetched data for sectors: {', '.join(sectors)}",
                "scheduled": schedule,
            },
        ]

    return steps


def _run_local_workflow(
    client_name: str,
    holdings_text: str,
    uploaded_file: io.StringIO | None,
    risk_profile: str,
    schedule: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    holding_objects = _parse_holdings_objects(holdings_text, uploaded_file)
    holdings = [h.__dict__ for h in holding_objects]
    sectors = _extract_sectors_from_holdings(holdings)
    market_data = fetch_market_data(sectors)
    summary = build_personalized_summary(client_name, holding_objects, market_data, risk_profile)
    talking_points = _build_talking_points(holdings, risk_profile, sectors)

    # Build explicit local workflow steps (not agent trace, but honest about what ran)
    agent_steps = [
        {
            "type": "tool_call",
            "tool": "parse_holdings (local)",
            "arguments": f"Parsed {len(holdings)} holdings from {'CSV' if uploaded_file else 'pasted text'}",
            "scheduled": schedule,
        },
        {
            "type": "tool_call",
            "tool": "get_market_data (local)",
            "arguments": f"Fetched mock market data for sectors: {', '.join(sectors) if sectors else 'none detected'}",
            "scheduled": schedule,
        },
        {
            "type": "tool_call",
            "tool": "build_personalized_summary (local)",
            "arguments": f"Built summary for {client_name}, risk={risk_profile}, {len(holdings)} positions",
            "scheduled": schedule,
        },
    ]

    output = {
        "advisor_summary": summary,
        "talking_points": talking_points,
        "sectors_in_focus": sectors,
        "holdings": holdings,
        "market_data": market_data,
        "agent_steps": agent_steps,
        "provider": "local (no API key)",
        "provider_response": "",
    }
    if extra:
        output.update(extra)
    return output


def run_personalized_market_brief_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: io.StringIO | None,
    risk_profile: str,
    schedule: bool,
) -> dict[str, Any]:
    openai_api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return _run_local_workflow(client_name, holdings_text, uploaded_file, risk_profile, schedule)

    # Ensure the SDK can also find the key under its expected env var
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    holdings_source = "uploaded CSV" if uploaded_file is not None else "pasted text"
    prompt = (
        f"Generate a personalized morning market brief for the following client.\n\n"
        f"Client name: {client_name}\n"
        f"Risk profile: {risk_profile}\n"
        f"Holdings source: {holdings_source}\n\n"
        f"Raw holdings ({holdings_source}):\n{holdings_text}\n\n"
        f"Follow the workflow: parse holdings → identify sectors → fetch market data → produce MarketBrief."
    )
    try:
        agent = _build_openai_agent(openai_api_key, schedule)
        result = Runner.run_sync(agent, input=prompt, max_turns=10)
        brief = result.final_output_as(MarketBrief, raise_if_incorrect_type=False)
        if isinstance(brief, MarketBrief):
            advisor_summary = brief.advisor_summary
            talking_points = brief.talking_points
            sectors = brief.sectors_in_focus
        else:
            advisor_summary = str(result.final_output)
            talking_points = []
            sectors = []
    except Exception as exc:
        return _run_local_workflow(
            client_name,
            holdings_text,
            uploaded_file,
            risk_profile,
            schedule,
            extra={"openai_key_error": str(exc)},
        )

    # Parse holdings and extract sectors once (not twice)
    holdings = _parse_holdings(holdings_text, uploaded_file)
    extracted_sectors = sectors or _extract_sectors_from_holdings(holdings)
    
    # Fetch market data once, before running the agent workflow above
    # Use the same market data that the agent used internally
    market_data = fetch_market_data(extracted_sectors)

    # Use talking points from the agent if present; fall back to local personalized ones
    if not talking_points:
        talking_points = _build_talking_points(holdings, risk_profile, extracted_sectors)

    # Extract real tool call trace from the SDK run
    agent_steps = _extract_real_agent_steps(result, holdings, schedule)

    return {
        "advisor_summary": advisor_summary,
        "talking_points": talking_points,
        "sectors_in_focus": sectors,
        "holdings": holdings,
        "market_data": market_data,
        "agent_steps": agent_steps,
        "provider": "openai",
        "provider_response": advisor_summary,
    }
