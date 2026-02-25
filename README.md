# Amazon Ads Automation (AI Agent Workflow)

This project uses an agentic workflow (via LangChain/LangGraph) to analyze Amazon Ads data, generate insights, and produce structured metrics reports.

## Overview

The system consists of several specialized agents:
- **Insights Agent**: Interprets tricky trends and qualitative data.
- **Metrics Agent**: Aggregates and calculates quantitative performance metrics.
- **Supervisor**: Coordinates the workflow and consolidates the final report.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Amazon-Ads-Automation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys.
   ```bash
   cp .env.example .env
   ```
   Required keys:
   - `ANTHROPIC_API_KEY` (for Claude models)

## Usage

Run the main script to start the analysis workflow:

```bash
python main.py --request "Generate an Amazon Ads performance report regarding Q3 ad spend efficiency."
```

**Optional Arguments:**
- `--start-date YYYY-MM-DD`: Filter data from this date.
- `--end-date YYYY-MM-DD`: Filter data up to this date.

## Output

- The script prints a structured JSON payload to the console.
- A file `metrics_output.json` is generated with the account summary.

## Project Structure

- `src/agents/`: Agent definitions (Insights, Metrics, Supervisor).
- `src/config/`: Configuration (LLM settings, etc.).
- `src/graph/`: LangGraph workflow definition.
- `src/tools/`: Tools for data loading and metric calculation.
- `data/`: Place your input Excel files here (e.g., `SD_AdvertisedProduct.xlsx`).
