import io
import os
import sys
import urllib.parse
from typing import Any, Dict

import streamlit as st
from openpyxl import Workbook

OPENAI_KEY_PRESENT = bool(os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY"))

# Ensure the project root is on sys.path when streamlit runs this file as a script.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from mili_market_brief_agent.agent import run_personalized_market_brief_agent
except ImportError:
    from agent import run_personalized_market_brief_agent


# Synthetic client profiles for testing
def build_excel_bytes(output: Dict[str, Any], client_name: str, risk_profile: str) -> bytes:
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
    summary_sheet.append(["Advisor Summary", ""])
    for line in output.get("advisor_summary", "").splitlines():
        summary_sheet.append([line])
    summary_sheet.append([])
    summary_sheet.append(["Provider Response", output.get("provider_response", "")])
    if output.get("openai_key_error"):
        summary_sheet.append(["OpenAI Key Error", output["openai_key_error"]])

    workflow_sheet = wb.create_sheet(title="Workflow")
    workflow_sheet.append(["Tool", "Description", "Result Count", "Sectors", "Scheduled"])
    for step in output.get("agent_steps", []):
        workflow_sheet.append([
            step.get("tool", ""),
            step.get("description", ""),
            step.get("result_count", ""),
            ", ".join(step.get("sectors", [])) if step.get("sectors") else "",
            step.get("scheduled", ""),
        ])

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


SAMPLE_CLIENTS = {
    "Margaret Chen (Conservative, Retired)": {
        "risk_profile": "Conservative",
        "holdings": "BND, 150, 18750, Bonds\nDVY, 75, 7650, Dividend Stocks\nJNJ, 30, 5400, Healthcare\nPEP, 25, 3750, Consumer Staples\nT, 100, 2300, Utilities",
    },
    "James Rodriguez (Moderate, Mid-Career)": {
        "risk_profile": "Moderate",
        "holdings": "VOO, 60, 32400, Broad Market\nQQQ, 40, 15200, Tech\nVEA, 50, 3500, International\nVNQ, 25, 3250, Real Estate\nBND, 30, 3750, Bonds",
    },
    "Sarah Thompson (Growth, Young Professional)": {
        "risk_profile": "Growth",
        "holdings": "AAPL, 100, 18000, Technology\nMSFT, 50, 13000, Technology\nGOOGL, 30, 3900, Technology\nNVDA, 25, 11250, Semiconductors\nAMZN, 15, 3000, Consumer Discretionary",
    },
    "Robert Williams (Aggressive, Entrepreneur)": {
        "risk_profile": "Growth",
        "holdings": "NVDA, 75, 33750, Technology\nTSLA, 40, 12000, Automotive\nCRWD, 50, 8000, Cybersecurity\nAI, 30, 2700, Artificial Intelligence\nARKK, 100, 5300, Innovation ETF",
    },
}


def main() -> None:
    st.set_page_config(page_title="Mili Market Brief Agent", layout="wide")

    # persist last generated output so downloads / reruns keep the summary visible
    if "last_output" not in st.session_state:
        st.session_state["last_output"] = None
        st.session_state["last_client"] = ""
        st.session_state["last_profile"] = ""

    st.markdown(
        """
        <style>
        .css-1f4mp12, .css-1avcm0n, label {
            color: #7C62C4 !important;
        }
        .main > div > .block-container h2,
        .main > div > .block-container h3 {
            color: #7C62C4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = os.path.join(os.path.dirname(__file__), "pics", "Mili Logo.svg")
    with open(logo_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    svg_data = urllib.parse.quote(svg_content)

    with st.container():
        st.markdown(
            f"""
            <div style='position:relative; margin-bottom:1rem; padding-top:0.5rem;'>
                <div style='position:absolute; top:0; left:0; z-index:10;'>
                    <img src='data:image/svg+xml;utf8,{svg_data}' style='width:75%; height:auto; max-height:none; display:block;' alt='Mili Logo' />
                </div>
            </div>
            <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:2rem; margin-bottom:1rem;'>
                <h2 style='color:#000000; text-align:center; margin:0.5rem 0 0.25rem;'>Mili Market Brief Agent</h2>
                <p style='text-align:center; margin:0;'>Use this demo agent to ingest a client's holdings, pull market data, and generate a personalized morning brief.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

    with st.sidebar.form(key="agent_input"):
        st.markdown("### Sample Clients")
        selected_client = st.selectbox(
            "Select a sample client profile",
            list(SAMPLE_CLIENTS.keys()),
            help="Choose a realistic client scenario to test with",
        )
        
        sample_data = SAMPLE_CLIENTS[selected_client]
        client_name = selected_client.split(" (")[0]
        risk_profile = st.selectbox("Client risk profile", ["Conservative", "Moderate", "Growth"], index=["Conservative", "Moderate", "Growth"].index(sample_data["risk_profile"]))
        
        if not OPENAI_KEY_PRESENT:
            st.warning("Set OPENAI_API_KEY in your environment to enable OpenAI Responses tool calls in the agent.")
        
        st.markdown("### Or Upload Custom Data")
        uploaded_file = st.file_uploader("Upload holdings CSV", type=["csv"])
        
        holdings_text = st.text_area(
            "Or paste holdings (ticker, quantity, market_value, sector)",
            sample_data["holdings"],
            height=200,
        )
        
        schedule = st.checkbox("Schedule for morning delivery", value=False)
        submit = st.form_submit_button("Generate Brief")

    if submit:
        csv_buffer = None
        if uploaded_file is not None:
            try:
                csv_bytes = uploaded_file.getvalue().decode("utf-8")
                csv_buffer = io.StringIO(csv_bytes)
            except Exception:
                st.error("Unable to read uploaded CSV file. Please check the format.")

        output = run_personalized_market_brief_agent(
            client_name=client_name,
            holdings_text=holdings_text,
            uploaded_file=csv_buffer,
            risk_profile=risk_profile,
            schedule=schedule,
        )

        # persist the latest generated report so downloads and reruns keep the view
        st.session_state["last_output"] = output
        st.session_state["last_client"] = client_name
        st.session_state["last_profile"] = risk_profile

    # Display the last generated report (if any)
    display_output = st.session_state.get("last_output")
    if display_output:
        if display_output.get("openai_key_error"):
            st.warning(
                "OpenAI key validation failed, so the app is using the local summary workflow instead. "
                f"Details: {display_output['openai_key_error']}"
            )

        # Summary container with download button aligned top-right
        with st.container():
            col_left, col_right = st.columns([8, 1])
            col_left.header("Advisor-ready Summary")
            # Show download only when a summary exists
            if display_output.get("advisor_summary"):
                excel_bytes = build_excel_bytes(display_output, st.session_state.get("last_client", ""), st.session_state.get("last_profile", ""))
                col_right.download_button(
                    label="Download",
                    data=excel_bytes,
                    file_name=f"{(st.session_state.get('last_client') or 'client').replace(' ', '_')}_market_brief.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{st.session_state.get('last_client')}",
                )

            col_left.markdown("```")
            col_left.text(display_output.get("advisor_summary", ""))
            col_left.markdown("```")

        with st.expander("Show structured agent output and tool reasoning"):
            st.json(display_output)

        with st.expander("Agent workflow steps"):
            for step in display_output.get("agent_steps", []):
                st.write(step)


if __name__ == "__main__":
    main()
