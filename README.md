# BizMind AI 🧠
> AI-powered Business Intelligence for Small Businesses

## Problem
Currently, over 63 million small businesses (kiranas, boutiques, cafés) in India have zero access to proper Business Intelligence (BI) tools. They record sales but lack actionable insights to optimize their business operations and increase profitability.

## Solution
BizMind AI is an end-to-end Generative AI platform that takes raw CSV sales data and transforms it into plain English (and Hindi) insights. With a conversational interface and smart alerts, any small business owner can act like they have a dedicated data analyst on staff.

## Architecture Diagram

```
+----------------+       +-------------------+       +-----------------------+
|                |       |                   |       |                       |
|   Sales CSV    +-------> Data Ingestion &  +------->    FAISS Vector       |
|   Upload       |       | Feature Eng.      |       |    Store (RAG)        |
|                |       |                   |       |                       |
+----------------+       +---------+---------+       +-----------+-----------+
                                   |                             |
                                   v                             v
+----------------+       +-------------------+       +-----------------------+
|                |       |                   |       |   LangChain Agent     |
|   Streamlit    | <-----+  LangChain Tools  | <-----+   (Groq LLaMA 3.70b)  |
|   Dashboard    |       |  (Anomalies, etc) |       |   & Conversation      |
|                |       |                   |       |                       |
+----------------+       +-------------------+       +-----------------------+
```

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-yellow.svg)

## Features
- **Data Ingestion & Cleaning**: Automatically handles missing values and adds derived features (e.g., day of week, revenue).
- **RAG-based Insight Generator**: Converts sales records into vector embeddings (FAISS) for conversational Q&A.
- **Smart Tools**: Includes anomaly detection, restock prediction, trend analysis, and revenue forecasting using linear regression.
- **Conversational Agent**: Chat with your data using Groq's LLaMA 3 in natural language.
- **Automated PDF Reports**: Click a button to generate a comprehensive weekly report for offline viewing.

## Demo
*(Add demo GIF here)*

## Setup & Run

1. **Clone the repository and install dependencies**:
   ```bash
   git clone <repo_url>
   cd bizzmind-ai
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Edit the `.env` file and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

3. **Generate Mock Data**:
   ```bash
   python data/generate_mock.py
   ```
   *(This creates `data/mock_store.csv`)*

4. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Sample Insights
- "Top product: Cooking Oil (₹45,200 revenue). Peak day: Saturday."
- "ANOMALY [HIGH]: Rice sales dropped 42% vs last month average."
- "Cooking Oil: ~3 days remaining. RESTOCK NOW."
- "Projected next week revenue: ₹28,400 (+12% vs this week)."

## Business Impact
- **Time saved**: Instant analysis instead of manual calculations.
- **Lost revenue prevented**: Restock alerts avoid out-of-stock scenarios.
- **Better marketing**: Knowing peak days and top products helps focus promotional efforts.

## Future Roadmap
- WhatsApp integration via Twilio
- Voice input in Hindi
- Multi-store support
- Mobile app integration
