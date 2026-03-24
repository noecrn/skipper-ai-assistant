# Skipper AI Assistant ⛵🤖

AI-powered sailing performance assistant for post-navigation analysis.

## Features
- **Data Ingestion**: Process raw telemetry (CSV) and calculate performance ratios using polar data.
- **Machine Learning Analysis**: Use XGBoost and SHAP to identify performance loss contributors.
- **Natural Language Explanations**: Generate human-readable diagnostics using a local LLM (Ollama).
- **Interactive Queries**: Ask specific questions about your navigation performance.

## Installation
Requires Python 3.10+ and [Ollama](https://ollama.com/) for LLM explanations.

```bash
pip install -e .
```

## CLI Usage

### 1. Ingest Data
```bash
skipper-ai ingest-data data/raw/your_run.csv --run-id my_navigation
```

### 2. Analyze Performance
```bash
skipper-ai analyze my_navigation
```

### 3. Generate Explanation
```bash
skipper-ai explain my_navigation
```

### 4. Ask Questions
```bash
skipper-ai ask my_navigation "Why was my performance low when upwind?"
```

### 5. HTML report (offline)
After ingest and analyze, build a single self-contained `report.html` (embedded charts and coach advice). If `explanation.txt` is missing, `report` will generate it (same as `explain`, requires Ollama unless the fallback applies).

```bash
skipper-ai report my_navigation
skipper-ai report my_navigation --open
skipper-ai report my_navigation --no-explain
```

Order: **ingest-data** → **analyze** → **report** (or run **explain** before **report** to avoid LLM work during reporting).

## Project Context
The project focuses on **explainability**. It uses SHAP to quantify how much each factor (heel, sail choice, wind conditions) contributed to the difference between actual boat speed and the theoretical polar speed.

---
Sources :
- http://www.regattagame.net/rg/rg2.0/zorglub/viewtopic.php?f=5&t=3399&sid=24aff1dbd6a6dc9f0fd5068b6471f1eb
