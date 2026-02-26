# Amazon Ads Automation (AI Agent Workflow)

This project leverages a multi-agent AI system based on LangChain and LangGraph to analyze Amazon Ads data. It automates the process of loading metrics, identifying trends, and generating strategic performance reports.

## Features

- **Multi-Agent Architecture**:
  - **Metrics Agent**: Computes granular performance metrics (CPC, ACOS, ROAS) from raw data.
  - **Insights Agent**: interpretation of metrics to provide qualitative strategic advice.
  - **Supervisor**: Orchestrates the workflow and delegates tasks between agents.
- **Workflow Automation**: Built on LangGraph to handle state and routing between analysis steps.
- **Structured Output**: Produces JSON-based metric bundles and structured insight reports.

## Structure

```
src/
├── agents/        # Agent definitions (Metrics, Insights, Supervisor)
├── config/        # Configuration and LLM setup
├── framework/     # Base classes for agents and registry
├── graph/         # LangGraph workflow construction
├── schemas/       # Pydantic models for data validation
└── tools/         # Data loading and calculation tools
```

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

4. **Environment Configuration:**
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Usage

Run the main analysis script:

```bash
python main.py --request "Analyze Q3 ad performance and suggest budget reallocations."
```

### Options

- `--request`: Description of the analysis task (default: "Generate an Amazon Ads performance report.").
- `--start-date`: Filter data starting from this date (YYYY-MM-DD).
- `--end-date`: Filter data up to this date (YYYY-MM-DD).

## Outputs

- `metrics_output.json`: Detailed quantitative metrics.
- Console Output: High-level insights and executive summary.
