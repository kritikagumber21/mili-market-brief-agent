import io
import os
import sys
import urllib.parse
from typing import Any, Dict

import streamlit as st

OPENAI_KEY_PRESENT = bool(os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY"))

# Ensure the project root is on sys.path when streamlit runs this file as a script.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from mili_market_brief_agent.agent import run_personalized_market_brief_agent
    from mili_market_brief_agent.tools import build_excel_bytes
except ImportError:
    from agent import run_personalized_market_brief_agent
    from tools import build_excel_bytes


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

# Map step types to readable labels and icons
STEP_TYPE_LABELS = {
    "tool_call": ("🔧", "Tool Called"),
    "tool_result": ("📤", "Tool Result"),
    "agent_message": ("💬", "Agent Message"),
}


def _render_agent_steps(steps: list[dict]) -> None:
    """Render agent steps in a structured, readable way."""
    if not steps:
        st.info("No agent steps recorded.")
        return

    for i, step in enumerate(steps, 1):
        step_type = step.get("type", "unknown")
        icon, label = STEP_TYPE_LABELS.get(step_type, ("•", step_type))

        if step_type == "tool_call":
            tool = step.get("tool", "unknown")
            args = step.get("arguments", "")
            st.markdown(f"**{icon} Step {i} — {label}: `{tool}`**")
            if args:
                st.code(args, language=None)

        elif step_type == "tool_result":
            preview = step.get("result_preview", "")
            st.markdown(f"**{icon} Step {i} — {label}**")
            if preview:
                st.code(preview, language="json")

        elif step_type == "agent_message":
            content = step.get("content", "")
            st.markdown(f"**{icon} Step {i} — {label}**")
            if content:
                st.markdown(f"> {content}")

        else:
            st.write(step)

        st.divider()


def main() -> None:
    st.set_page_config(page_title="Mili Market Brief Agent", layout="wide")

    if "last_output" not in st.session_state:
        st.session_state["last_output"] = None
        st.session_state["last_client"] = ""
        st.session_state["last_profile"] = ""

    logo_path = os.path.join(os.path.dirname(__file__), "pics", "Mili Logo.svg")
    try:
        with open(logo_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        svg_data = urllib.parse.quote(svg_content)
        logo_html = f"<img src='data:image/svg+xml;utf8,{svg_data}' style='width:75%; height:auto; display:block;' alt='Mili Logo' />"
    except FileNotFoundError:
        logo_html = ""

    with st.container():
        st.markdown(
            f"""
            <div style='position:relative; margin-bottom:1rem; padding-top:0.5rem;'>
                <div style='position:absolute; top:0; left:0; z-index:10;'>{logo_html}</div>
            </div>
            <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:2rem; margin-bottom:1rem;'>
                <h2 style='color:#000000; text-align:center; margin:0.5rem 0 0.25rem;'>Mili Market Brief Agent</h2>
                <p style='text-align:center; margin:0;'>Ingest a client's holdings, pull market data, and generate a personalized morning brief.</p>
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
        risk_profile = st.selectbox(
            "Client risk profile",
            ["Conservative", "Moderate", "Growth"],
            index=["Conservative", "Moderate", "Growth"].index(sample_data["risk_profile"]),
        )

        if not OPENAI_KEY_PRESENT:
            st.error(
                "⚠️ **OpenAI API key required.** "
                "Please set the `AI_API_KEY` or `OPENAI_API_KEY` environment variable to use this agent."
            )

        st.markdown("### Or Upload Custom Data")
        uploaded_file = st.file_uploader("Upload holdings CSV", type=["csv"])

        holdings_text = st.text_area(
            "Or paste holdings (ticker, quantity, market_value, sector)",
            sample_data["holdings"],
            height=200,
        )

        schedule = st.checkbox("Schedule for morning delivery", value=False)
        submit = st.form_submit_button("Generate Brief", disabled=not OPENAI_KEY_PRESENT)

    if submit:
        # Input validation
        if not holdings_text or not holdings_text.strip():
            st.error("⚠️ Please provide holdings data (upload CSV or paste holdings).")
        else:
            csv_buffer = None
            if uploaded_file is not None:
                try:
                    csv_bytes = uploaded_file.getvalue().decode("utf-8")
                    csv_buffer = io.StringIO(csv_bytes)
                except Exception:
                    st.error("Unable to read uploaded CSV file. Please check the format.")

            try:
                with st.spinner("Running agent…"):
                    output = run_personalized_market_brief_agent(
                        client_name=client_name,
                        holdings_text=holdings_text,
                        uploaded_file=csv_buffer,
                        risk_profile=risk_profile,
                        schedule=schedule,
                    )
            except ValueError as e:
                st.error(f"Agent error: {str(e)}")
                return

            st.session_state["last_output"] = output
            st.session_state["last_client"] = client_name
            st.session_state["last_profile"] = risk_profile

    display_output = st.session_state.get("last_output")
    if display_output:
        provider = display_output.get("provider", "")
        if provider:
            st.caption(f"Provider: **{provider}**")

        # ── Advisor summary ──────────────────────────────────────────────────
        with st.container():
            col_left, col_right = st.columns([8, 1])
            col_left.header("Advisor-ready Summary")
            if display_output.get("advisor_summary"):
                excel_bytes = build_excel_bytes(
                    display_output,
                    st.session_state.get("last_client", ""),
                    st.session_state.get("last_profile", ""),
                )
                col_right.download_button(
                    label="Download",
                    data=excel_bytes,
                    file_name=f"{(st.session_state.get('last_client') or 'client').replace(' ', '_')}_market_brief.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{st.session_state.get('last_client')}",
                )
            # Display full summary with proper text wrapping (not code block)
            col_left.write(display_output.get("advisor_summary", ""))

        # ── Talking points ───────────────────────────────────────────────────
        talking_points = display_output.get("talking_points", [])
        if talking_points:
            st.subheader("Talking Points")
            for point in talking_points:
                st.markdown(f"- {point}")

        # ── Agent reasoning trace ────────────────────────────────────────────
        with st.expander("Agent reasoning trace (tool calls & results)", expanded=False):
            _render_agent_steps(display_output.get("agent_steps", []))

        # ── Full structured output ───────────────────────────────────────────
        with st.expander("Full structured output (JSON)", expanded=False):
            st.json(display_output)


if __name__ == "__main__":
    main()
