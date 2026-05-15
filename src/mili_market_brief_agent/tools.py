import csv
import io
from dataclasses import dataclass
from typing import Any, Dict, List

from openpyxl import Workbook


@dataclass
class ClientHolding:
    ticker: str
    quantity: float
    market_value: float
    sector: str


def parse_holdings_text(raw_text: str) -> List[ClientHolding]:
    """Parse simple holdings text into a structured list."""
    holdings = []
    for line in raw_text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 3:
            continue
        ticker = parts[0].upper()
        try:
            quantity = float(parts[1])
        except ValueError:
            quantity = 0.0
        try:
            market_value = float(parts[2])
        except ValueError:
            market_value = 0.0
        sector = parts[3] if len(parts) > 3 else "Unknown"
        holdings.append(ClientHolding(ticker=ticker, quantity=quantity, market_value=market_value, sector=sector))
    return holdings


def parse_holdings_csv(uploaded_csv: io.StringIO) -> List[ClientHolding]:
    reader = csv.DictReader(uploaded_csv)
    holdings = []
    for row in reader:
        ticker = row.get("ticker", row.get("symbol", "")).upper()
        if not ticker:
            continue
        try:
            quantity = float(row.get("quantity", row.get("shares", 0)))
        except (ValueError, TypeError):
            quantity = 0.0
        try:
            market_value = float(row.get("market_value", row.get("value", 0)))
        except (ValueError, TypeError):
            market_value = 0.0
        sector = row.get("sector", "Unknown")
        holdings.append(ClientHolding(ticker=ticker, quantity=quantity, market_value=market_value, sector=sector))
    return holdings


def fetch_market_data(sectors: List[str]) -> Dict[str, Any]:
    """Return mocked market data and news for the sectors referenced by the client."""
    # Sector-specific movers map
    sector_movers_map = {
        "Technology": [
            {"ticker": "AAPL", "move": "+2.2%", "reason": "strong earnings reaction"},
            {"ticker": "MSFT", "move": "+1.1%", "reason": "AI spending commentary"},
            {"ticker": "NVDA", "move": "+1.8%", "reason": "data center demand momentum"},
        ],
        "Tech": [
            {"ticker": "AAPL", "move": "+2.2%", "reason": "strong earnings reaction"},
            {"ticker": "MSFT", "move": "+1.1%", "reason": "AI spending commentary"},
            {"ticker": "NVDA", "move": "+1.8%", "reason": "data center demand momentum"},
        ],
        "Energy": [
            {"ticker": "XOM", "move": "+1.5%", "reason": "oil price stabilization"},
            {"ticker": "CVX", "move": "+1.2%", "reason": "upstream guidance beat"},
            {"ticker": "COP", "move": "+0.9%", "reason": "production growth expectations"},
        ],
        "Financials": [
            {"ticker": "JPM", "move": "+0.8%", "reason": "net interest margin resilience"},
            {"ticker": "BAC", "move": "+0.6%", "reason": "deposit stability signals"},
            {"ticker": "GS", "move": "+1.1%", "reason": "investment banking pipeline strength"},
        ],
        "Healthcare": [
            {"ticker": "JNJ", "move": "+0.5%", "reason": "drug approval pipeline confidence"},
            {"ticker": "PFE", "move": "+1.3%", "reason": "vaccine revenue growth"},
            {"ticker": "UNH", "move": "+0.9%", "reason": "healthcare utilization trends"},
        ],
        "Consumer Discretionary": [
            {"ticker": "AMZN", "move": "+1.6%", "reason": "e-commerce and ad revenue strength"},
            {"ticker": "NKE", "move": "-0.4%", "reason": "wholesale channel pressure"},
            {"ticker": "MCD", "move": "+0.3%", "reason": "consumer traffic resilience"},
        ],
        "Consumer Staples": [
            {"ticker": "PEP", "move": "+0.4%", "reason": "pricing power maintained"},
            {"ticker": "JNJ", "move": "+0.5%", "reason": "consumer health demand steady"},
            {"ticker": "KO", "move": "+0.2%", "reason": "volume trends stable"},
        ],
        "Bonds": [
            {"ticker": "BND", "move": "+0.8%", "reason": "yield curve flattening benefit"},
            {"ticker": "AGG", "move": "+0.7%", "reason": "credit spread compression"},
            {"ticker": "LQD", "move": "+0.5%", "reason": "investment-grade resilience"},
        ],
        "Dividend Stocks": [
            {"ticker": "DVY", "move": "+0.6%", "reason": "dividend aristocrat strength"},
            {"ticker": "T", "move": "+0.3%", "reason": "telecom dividend stability"},
            {"ticker": "DIS", "move": "+1.1%", "reason": "streaming segment momentum"},
        ],
        "Utilities": [
            {"ticker": "T", "move": "+0.3%", "reason": "rate stability expectations"},
            {"ticker": "NEE", "move": "+0.9%", "reason": "renewable energy tailwinds"},
            {"ticker": "DUK", "move": "+0.4%", "reason": "grid modernization demand"},
        ],
    }

    # Headline templates by sector
    sector_headlines = {
        "Technology": "Tech shares remain in focus after the earnings season.",
        "Tech": "Tech shares remain in focus after the earnings season.",
        "Energy": "Energy names are reacting to oil price stabilization.",
        "Financials": "Bank earnings are being watched for margin commentary.",
        "Healthcare": "Healthcare sector benefiting from pharma innovation optimism.",
        "Consumer Discretionary": "Discretionary names showing resilience amid consumer spending data.",
        "Consumer Staples": "Staple stocks stable as pricing power and volume hold steady.",
        "Bonds": "Fixed income benefiting from improved economic data.",
        "Dividend Stocks": "Dividend-paying equities attractive in current yield environment.",
        "Utilities": "Utilities supported by rate stability and energy transition focus.",
    }

    # Build personalized movers from relevant sectors
    top_movers = []
    for sector in sectors:
        if sector in sector_movers_map:
            top_movers.extend(sector_movers_map[sector])
    
    # Deduplicate by ticker and limit to top 3
    seen_tickers = set()
    unique_movers = []
    for mover in top_movers:
        if mover["ticker"] not in seen_tickers:
            unique_movers.append(mover)
            seen_tickers.add(mover["ticker"])
    top_movers = unique_movers[:3]

    # If no sector-specific movers found, use balanced defaults
    if not top_movers:
        top_movers = [
            {"ticker": "AAPL", "move": "+2.2%", "reason": "strong earnings reaction"},
            {"ticker": "TSLA", "move": "-1.4%", "reason": "autonomy guidance pressure"},
            {"ticker": "MSFT", "move": "+1.1%", "reason": "AI spending commentary"},
        ]

    # Build headlines
    headlines = ["Equities rallied after U.S. inflation data came in cooler than expected."]
    for sector in sectors:
        if sector in sector_headlines:
            headlines.append(sector_headlines[sector])

    return {
        "top_movers": top_movers,
        "headlines": headlines,
        "sector_coverage": list({s for s in sectors if s != "Unknown"})[:3],
    }


def build_personalized_summary(
    client_name: str,
    holdings: List[ClientHolding],
    market_data: Dict[str, Any],
    risk_profile: str,
) -> str:
    """Construct an advisor-ready personalized market brief with client-specific insights."""
    if not holdings:
        return "No client holdings were provided. Please upload or paste holdings data to generate a brief."

    primary_sectors = list({h.sector for h in holdings if h.sector and h.sector != "Unknown"})
    total_value = sum(h.market_value for h in holdings)
    top_assets = sorted(holdings, key=lambda h: h.market_value, reverse=True)[:3]
    top_tickers = ", ".join(a.ticker for a in top_assets)
    sector_str = ", ".join(primary_sectors) if primary_sectors else "diversified sectors"
    risk_label = risk_profile.lower()

    lines = [
        f"Morning Brief — {client_name or 'Client'}",
        f"Risk profile: {risk_profile}  |  Portfolio value: ${total_value:,.0f}",
        "=" * 60,
        "",
        "MARKET SNAPSHOT",
    ]

    for mover in market_data["top_movers"]:
        lines.append(f"  {mover['ticker']:6s} {mover['move']:>7s}  — {mover['reason']}")

    lines += [
        "",
        "WHAT THIS MEANS FOR THE CLIENT",
        f"  {client_name or 'The client'} holds a {risk_label}-risk portfolio concentrated in {sector_str}.",
        f"  Largest positions are {top_tickers}, which together represent the bulk of market value exposure.",
    ]

    for headline in market_data.get("headlines", []):
        lines.append(f"  • {headline}")

    # Risk-profile-specific framing
    if risk_profile == "Conservative":
        risk_note = "Focus on capital preservation — flag any outsized moves in fixed income or dividend names."
    elif risk_profile == "Moderate":
        risk_note = "Monitor sector concentration; consider whether recent volatility warrants any tactical trim."
    else:  # Growth / Aggressive
        risk_note = "Growth-oriented positioning may amplify gains and losses — ensure client is aligned on drawdown tolerance."

    lines += [
        "",
        "ADVISOR TALKING POINTS",
        f"  1. {risk_note}",
        f"  2. Check whether {top_tickers} exposure is still within target weight given today's moves.",
        f"  3. Ask the client if any upcoming liquidity needs should inform near-term positioning.",
        "",
        "Prepared by Mili Market Brief Agent.",
    ]

    if not any(s for s in primary_sectors if s):
        lines.append("Note: Sector data was incomplete — review holdings for missing sector tags.")

    return "\n".join(lines)


def build_json_output(holdings: List[ClientHolding], market_data: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "holdings": [holding.__dict__ for holding in holdings],
        "market_data": market_data,
        "agent_steps": steps,
    }


def build_excel_bytes(output: Dict[str, Any], client_name: str, risk_profile: str) -> bytes:
    """Generate an Excel workbook from the market brief output."""
    wb = Workbook()
    holdings_sheet = wb.active
    holdings_sheet.title = "Holdings"
    holdings_sheet.append(["Client Name", "Risk Profile", "Ticker", "Quantity", "Market Value", "Sector"])
    for holding in output.get("holdings", []):
        holdings_sheet.append([
            client_name,
            risk_profile,
            holding.get("ticker", ""),
            holding.get("quantity", 0),
            holding.get("market_value", 0),
            holding.get("sector", ""),
        ])

    summary_sheet = wb.create_sheet(title="Summary")
    summary_sheet.append(["Client Name", client_name or ""])
    summary_sheet.append(["Risk Profile", risk_profile or ""])
    summary_sheet.append(["Provider", output.get("provider", "")])
    summary_sheet.append(["Advisor Summary", ""])
    for line in output.get("advisor_summary", "").splitlines():
        summary_sheet.append([line])
    summary_sheet.append([])
    summary_sheet.append(["Talking Points", ""])
    for point in output.get("talking_points", []):
        summary_sheet.append([point])

    workflow_sheet = wb.create_sheet(title="Workflow")
    workflow_sheet.append(["Step", "Type", "Tool", "Arguments / Content", "Scheduled"])
    for i, step in enumerate(output.get("agent_steps", []), 1):
        step_type = step.get("type", "")
        tool = step.get("tool", "")
        content = step.get("arguments", step.get("result_preview", step.get("content", "")))
        workflow_sheet.append([i, step_type, tool, content, step.get("scheduled", "")])

    market_sheet = wb.create_sheet(title="Market Data")
    market_sheet.append(["Top Movers"])
    for mover in output.get("market_data", {}).get("top_movers", []):
        market_sheet.append([f"{mover.get('ticker', '')}: {mover.get('move', '')} ({mover.get('reason', '')})"])
    market_sheet.append([])
    market_sheet.append(["Headlines"])
    for headline in output.get("market_data", {}).get("headlines", []):
        market_sheet.append([headline])

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.read()
