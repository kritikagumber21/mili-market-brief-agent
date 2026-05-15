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
            "specifically to this client given their holdings and risk profile, (c) one clear action "
            "item or question the advisor should raise. Mention scheduled delivery only if schedule=True.\n"
            "- talking_points: 3–4 punchy bullet points the advisor can reference during a call. "
            "Each should be specific to this client's positions — not generic market commentary.\n"
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


def _build_talking_points() -> list[str]:
    return [
        "Confirm whether the client is comfortable with today's sector drivers and any recent performance volatility.",
        "Discuss whether the allocation remains aligned with the client’s time horizon and risk preferences.",
        "Consider whether any rebalancing or risk reduction is needed in high-concentration positions.",
    ]


def _build_agent_steps(
    holdings: list[dict],
    market_data: dict,
    provider: str,
    schedule: bool,
) -> list[dict]:
    sectors = _extract_sectors_from_holdings(holdings)
    return [
        {
            "tool": "Client Holdings Parser",
            "description": "Parsed the client's raw holdings into structured positions.",
            "result_count": len(holdings),
            "sectors": sectors,
            "scheduled": schedule,
        },
        {
            "tool": "Market Data and News Fetcher",
            "description": "Fetched market data and sector headlines for the client's holdings.",
            "result_count": len(market_data.get("top_movers", [])),
            "sectors": sectors,
            "scheduled": schedule,
        },
        {
            "tool": "Summary Builder",
            "description": "Built the advisor-ready summary based on holdings, risk profile, and market context.",
            "result_count": len(_build_talking_points()),
            "sectors": sectors,
            "scheduled": schedule,
        },
    ]



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
    output = {
        "advisor_summary": summary,
        "talking_points": _build_talking_points(),
        "sectors_in_focus": sectors,
        "holdings": holdings,
        "market_data": market_data,
        "agent_steps": _build_agent_steps(holdings, market_data, provider="local", schedule=schedule),
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

    # Ensure the SDK can also find the key
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

    holdings_source = "uploaded CSV" if uploaded_file is not None else "pasted text"
    prompt = (
        f"Generate a personalized morning market brief for the following client.\n\n"
        f"Client name: {client_name}\n"
        f"Risk profile: {risk_profile}\n"
        f"Morning delivery scheduled: {'Yes' if schedule else 'No'}\n"
        f"Holdings source: {holdings_source}\n\n"
        f"Raw holdings ({holdings_source}):\n{holdings_text}\n\n"
        f"Follow the workflow: parse holdings → identify sectors → fetch market data → produce MarketBrief."
    )
    try:
        agent = _build_openai_agent(openai_api_key)
        result = Runner.run_sync(agent, input=prompt, max_turns=10)
        brief = result.final_output_as(MarketBrief, raise_if_incorrect_type=False)
        if isinstance(brief, MarketBrief):
            advisor_summary = brief.advisor_summary
            talking_points = brief.talking_points
            sectors = brief.sectors_in_focus
        else:
            advisor_summary = str(result.final_output)
            talking_points = _build_talking_points()
            sectors = []
    except Exception as exc:
        return _run_local_workflow(
            client_name,
            holdings_text,
            uploaded_file,
            risk_profile,
            schedule,
            extra={"openai_key_error": str(exc), "openai_response": ""},
        )

    holdings = _parse_holdings(holdings_text, uploaded_file)
    market_data = fetch_market_data(sectors or _extract_sectors_from_holdings(holdings))
    return {
        "advisor_summary": advisor_summary,
        "talking_points": talking_points,
        "sectors_in_focus": sectors,
        "holdings": holdings,
        "market_data": market_data,
        "agent_steps": _build_agent_steps(holdings, market_data, provider="openai", schedule=schedule),
        "provider_response": "",
        "openai_response": "",
    }
