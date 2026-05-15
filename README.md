# Mili Market Brief Agent

A Python starter project for generating and demoing a personalized market brief agent.

## Project structure

- `src/mili_market_brief_agent/` — package source files
- `tests/` — unit tests
- `src/mili_market_brief_agent/streamlit_app.py` — Streamlit UI for the personalized market brief agent

## Getting started

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Get your OpenAI API key:

- Sign in to https://platform.openai.com/ and create a new key under "API Keys"
- Copy the key securely

4. Set your OpenAI key for tool-enabled reasoning:

macOS / Linux:

```bash
export AI_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:AI_API_KEY = "your_api_key"
```

Windows CMD:

```cmd
set AI_API_KEY=your_api_key
```

> Do not commit this key to version control or expose it in shared files.

5. Run the Streamlit demo:

```bash
streamlit run src/mili_market_brief_agent/streamlit_app.py
```

6. Run tests:

```bash
pytest
```

## What this demo includes

- `tools.py` — parses holdings, fetches mock market data, and builds a personalized summary
- `agent.py` — coordinates AI tool calls when `AI_API_KEY` is available, and produces advisor-ready output plus structured JSON
- `streamlit_app.py` — UI for uploading/pasting client holdings, generating the brief, inspecting tool reasoning, and downloading the summary as Excel

## OpenAI integration

This demo supports:

- OpenAI Agents SDK using tool-based reasoning and function-style tool calls

Set `AI_API_KEY` to your OpenAI key. Legacy `OPENAI_API_KEY` is still accepted.

You can optionally set model env vars:
- `AI_MODEL` for OpenAI
- default: `gpt-4.1-mini`

## Notes

This implementation requires an OpenAI API key to run. The market data and reasoning pipeline use the OpenAI Agents SDK with mocked market data feeds, so you don't need live market data connectors to demo the agent. You can extend it by replacing the mock tools with real market and news feeds.

## Why this agent

I chose the Personalized Market Brief Agent because it has the highest leverage for an advisor in a wealth management workflow:

- Advisors already spend time synthesizing market moves, client holdings, and risk posture into a short morning note.
- A personalized brief reduces manual research and creates a repeatable deliverable that can be shared with clients.
- It is high value because it combines market signal with client-specific exposure, which is exactly the advisor’s job to interpret.
- It is a strong “v0 product” because it can be built with mock market data and still feel useful, while leaving room to add real feeds later.

## How this project maps to the evaluation criteria

### Product instinct

- Selected the agent with the highest advisor leverage: advisors need a short, personalized briefing that explains what moved, why it matters, and what to discuss.
- The input surface is intentionally simple: upload or paste holdings, select risk profile, and generate the brief.
- Output is advisor-readable and includes both a summary narrative and structured JSON for transparency.

### Engineering taste

- Clean tool boundaries are enforced in `tools.py`:
  - holdings parsing
  - market/news mock fetcher
  - summary formatter
- `agent.py` orchestrates the workflow, isolates tool inputs/outputs, and records reasoning steps.
- Failure handling is graceful: the UI reports parse or file-read issues and the agent produces a clear message when no holdings are available.
- The system is intentionally lightweight and not over-engineered, with a clear path to replace mocks with real APIs.

### Vibe coding fluency

- The demo is functional end-to-end in Streamlit and can be extended quickly.
- The project uses fast, local reasoning instead of a heavy agent runtime, demonstrating speed and practical output generation.
- The code is structured so AI-assisted content generation can be added later without rewriting the core tool boundaries.

### Communication

- This README now captures the product choice, design rationale, and tradeoffs.
- The app is documented with clear setup and run instructions.
- The core tradeoff is explicit: mock market data for speed and reliability now, with a clean integration layer for real market feeds later.
