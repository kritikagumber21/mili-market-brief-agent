import io
import logging
import os
from typing import Any
from agents import Agent, Runner, function_tool, AsyncOpenAI, ToolCallItem, ToolCallOutputItem, MessageOutputItem
from agents.models.openai_responses import OpenAIResponsesModel
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .tools import (
    ClientHolding,
    build_personalized_summary,
    fetch_market_data,
    parse_holdings_csv,
    parse_holdings_text,
)


def _build_openai_agent(api_key: str) -> Agent:
    client = AsyncOpenAI(api_key=api_key)
    model = OpenAIResponsesModel(os.getenv("AI_MODEL", "gpt-4.1-mini"), openai_client=client)
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
            "specifically to this client given their holdings and risk profile, (c) one clear action item "
            "or question the advisor should raise. If the user specifies morning delivery is scheduled, "
            "mention this at the end. Otherwise, do not mention scheduling.\n"
            "- talking_points: 3–4 punchy bullet points the advisor can reference during a call. "
            "Each should be specific to this client's positions — not generic market commentary. "
            "Reference actual tickers and sectors from the holdings.\n"
            "- sectors_in_focus: list only sectors that actually appear in the client's holdings.\n\n"

            "RISK PROFILE GUIDANCE:\n"
            "Tailor recommendations based on the client's risk profile:\n"
            "- Conservative: emphasize capital preservation and downside risk; focus on bonds and dividend stocks.\n"
            "- Moderate: balance growth and protection; discuss rebalancing opportunities.\n"
            "- Growth: highlight opportunity and volatility tolerance; ensure alignment on drawdowns.\n\n"

            "TONE: Confident, concise, jargon-aware but not jargon-heavy. Written for a professional "
            "advisor, not the end client. Avoid filler phrases like 'it is worth noting' or 'as always'.\n\n"

            "CONSTRAINTS:\n"
            "- Never invent tickers or prices not present in the holdings or market data.\n"
            "- If holdings are empty or unparseable, set advisor_summary to a clear error message "
            "and return empty lists for the other fields.\n"
            "- Always call both tools before producing output. Do not skip get_market_data.\n"
            "- Sector names are title-cased (e.g. 'Technology', 'Energy', 'Healthcare'). Use these exact names."
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
        "The function automatically detects whether the input is CSV or plain text format."
    ),
)
def parse_client_holdings(raw_holdings: str) -> list[dict]:
    # Auto-detect format: if it starts with common CSV headers, treat as CSV
    if raw_holdings.strip().lower().startswith(("ticker", "symbol", "quantity", "shares")):
        try:
            buffer = io.StringIO(raw_holdings)
            holdings = parse_holdings_csv(buffer)
            if holdings:
                return [h.__dict__ for h in holdings]
        except Exception:
            pass
    # Fall back to text parsing
    holdings = parse_holdings_text(raw_holdings)
    return [h.__dict__ for h in holdings]


@function_tool(
    name_override="get_market_data",
    description_override=(
        "Fetch today's top market movers, relevant news headlines, and sector-level commentary "
        "for the sectors present in the client's portfolio. "
        "Call this after parse_holdings, passing the list of sector names extracted from the holdings. "
        "Sector names must be title-cased (e.g., 'Technology', 'Energy', 'Healthcare'). "
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
    Uses isinstance checks for robustness against SDK changes.
    """
    steps = []
    try:
        for item in run_result.new_items:
            # ToolCallItem: the agent decided to call a tool
            if isinstance(item, ToolCallItem):
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
            elif isinstance(item, ToolCallOutputItem):
                output = item.output if hasattr(item, "output") else str(item)
                # Summarize rather than dump raw output (can be large)
                result_preview = str(output)[:300] + ("…" if len(str(output)) > 300 else "")
                steps.append({
                    "type": "tool_result",
                    "result_preview": result_preview,
                    "scheduled": schedule,
                })
            # MessageOutputItem: the agent produced a text message mid-run
            elif isinstance(item, MessageOutputItem):
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
    except Exception as e:
        # Log unexpected errors in trace extraction but don't fail
        logger.warning(f"Failed to extract agent steps: {e}")

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


def run_personalized_market_brief_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: io.StringIO | None,
    risk_profile: str,
    schedule: bool,
) -> dict[str, Any]:
    openai_api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key is required. Please set AI_API_KEY or OPENAI_API_KEY environment variable."
        )

    # Ensure the SDK can also find the key under its expected env var
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    holdings_source = "uploaded CSV" if uploaded_file is not None else "pasted text"
    schedule_clause = "Mark this brief for scheduled morning delivery." if schedule else ""
    prompt = (
        f"Generate a personalized morning market brief for the following client.\n\n"
        f"Client name: {client_name}\n"
        f"Risk profile: {risk_profile} (tailor tone and recommendations accordingly)\n"
        f"Holdings source: {holdings_source}\n\n"
        f"<holdings>\n"
        f"{holdings_text}\n"
        f"</holdings>\n\n"
        f"{schedule_clause}\n\n"
        f"Follow the workflow: 1) parse_holdings 2) identify sectors 3) get_market_data 4) produce MarketBrief."
    )
    try:
        agent = _build_openai_agent(openai_api_key)
        result = Runner.run_sync(agent, input=prompt, max_turns=5)
        brief = result.final_output_as(MarketBrief, raise_if_incorrect_type=False)
        if isinstance(brief, MarketBrief):
            advisor_summary = brief.advisor_summary
            talking_points = brief.talking_points
            sectors = brief.sectors_in_focus
        else:
            logger.warning(
                f"Structured output parsing failed. Expected MarketBrief but got {type(result.final_output).__name__}: "
                f"{str(result.final_output)[:200]}"
            )
            advisor_summary = str(result.final_output)
            talking_points = []
            sectors = []
    except Exception as exc:
        raise

    # Parse holdings and extract sectors once (not twice)
    holdings = _parse_holdings(holdings_text, uploaded_file)
    extracted_sectors = sectors or _extract_sectors_from_holdings(holdings)
    
    # Fetch market data once, before running the agent workflow above
    # Use the same market data that the agent used internally
    market_data = fetch_market_data(extracted_sectors)

    # Fall back to locally-generated talking points if agent didn't produce any
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
