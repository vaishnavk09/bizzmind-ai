# BizMind AI

> AI-powered business intelligence for small shop owners in India.

Upload your daily sales CSV and ask plain-English (or Hindi) questions like:
"Which product is about to run out?" or "What was my revenue last week?"

---

## Architecture

```
CSV Upload / Demo Data
        │
        ▼
DataIngestionPipeline          ← clean, feature-engineer, chunk
        │                ╲
        │                 FAISS Index + BM25 Corpus
        │                         │
        ▼                         ▼
  DataFrame (pandas)       VectorStoreManager
        │                 (Hybrid RAG: RRF fusion)
        └────────┬────────────────┘
                 │
           ToolContext (injected — no globals)
                 │
         ┌───────┼───────────────────────┐
         │       │                       │
   analyze_  detect_  predict_   forecast_  generate_
   trends  anomalies  restock    revenue    insight
                 │
         BizMindAgent  (LangGraph ReAct, Groq Llama 3.3-70b)
                 │
        Streamlit UI  (Dashboard / Chat / Alerts / Report)
```

---

## Quickstart

```bash
# 1. Clone and enter the project
cd bizzmind-ai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (https://console.groq.com)

# 5. (Optional) Generate demo data
python data/generate_mock.py

# 6. Launch the app
streamlit run app.py
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Hybrid RAG** | FAISS dense search + BM25 sparse search, fused via Reciprocal Rank Fusion |
| **ReAct Agent** | LangGraph agent with 5 specialist tools and few-shot system prompt |
| **Revenue Forecasting** | Linear trend + 90% bootstrap confidence interval |
| **Anomaly Detection** | Rolling z-score + percentage change, severity-ranked |
| **Restock Predictor** | Sales-velocity-based stockout estimation |
| **Tool Trace Panel** | Every agent response shows which tools were called and why |
| **Demo Mode** | One-click dataset load — no upload required |
| **RAG Evaluation** | Offline golden-set evaluator (context recall + faithfulness) |
| **Multi-language** | English and Hindi support |
| **PDF Reports** | Auto-generated weekly summary with AI insights |

---

## RAG Evaluation

```bash
python -m eval.rag_eval
# Outputs eval/eval_results.json with per-question pass/fail and aggregate scores
```

---

## Project Structure

```
bizzmind-ai/
├── app.py                      # Streamlit entrypoint
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── agent.py                # BizMindAgent (LangGraph ReAct)
│   ├── ingestion.py            # DataIngestionPipeline
│   ├── vector_store.py         # Hybrid RAG (FAISS + BM25 + RRF)
│   ├── report.py               # PDF report generation
│   └── tools/
│       ├── base.py             # ToolContext dataclass
│       ├── trend_analyzer.py
│       ├── anomaly_detector.py
│       ├── restock_predictor.py
│       ├── revenue_forecaster.py
│       └── insight_generator.py
├── eval/
│   └── rag_eval.py             # Offline RAG evaluation suite
└── data/
    └── generate_mock.py        # Mock store data generator
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | From https://console.groq.com |
| `LANGCHAIN_TRACING_V2` | Optional | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | Optional | LangSmith key |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |
