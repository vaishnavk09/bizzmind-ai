import os
import uuid
import logging
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from pipeline.ingestion import DataIngestionPipeline
from pipeline.vector_store import VectorStoreManager
from pipeline.agent import BizMindAgent
from pipeline.report import generate_weekly_report
from pipeline.tools.anomaly_detector import detect_anomalies, set_context_df as set_anom
from pipeline.tools.restock_predictor import predict_restock, set_context_df as set_rest
from pipeline.tools.revenue_forecaster import forecast_revenue, set_context_df as set_rev

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bizmind.app")

st.set_page_config(
    page_title="BizMind AI",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------

defaults = {
    "df": None,
    "agent": None,
    "chat_history": [],
    "metrics": None,
    "language": "English",
    "session_id": str(uuid.uuid4()),  # unique per browser tab / session
    "tool_traces": [],                # list of [{tool, output}] per assistant message
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ------------------------------------------------------------------
# Data processing helpers
# ------------------------------------------------------------------

def _build_pipeline(csv_path: str):
    pipeline = DataIngestionPipeline()
    df = pipeline.load_csv(csv_path)
    df = pipeline.clean_data(df)
    df = pipeline.add_features(df)
    metrics = pipeline.get_summary_stats(df)
    chunks = pipeline.to_text_chunks(df)

    vsm = VectorStoreManager(index_path="faiss_index")
    vsm.build_store(chunks)

    agent = BizMindAgent(df, vsm)
    return df, metrics, agent


def process_uploaded_file(file):
    with st.spinner("Processing data…"):
        # Write to a proper temp file — never the CWD
        suffix = os.path.splitext(file.name)[1] or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        try:
            df, metrics, agent = _build_pipeline(tmp_path)
            st.session_state.df = df
            st.session_state.metrics = metrics
            st.session_state.agent = agent
            st.session_state.chat_history = []
            st.session_state.tool_traces = []
            logger.info("Uploaded file processed. Rows: %d", len(df))
            st.success("✅ Data processed successfully!")
        finally:
            os.unlink(tmp_path)


def load_demo():
    with st.spinner("Loading demo dataset…"):
        demo_path = os.path.join(os.path.dirname(__file__), "data", "mock_store.csv")
        if not os.path.exists(demo_path):
            # Generate it on-the-fly if missing
            from data.generate_mock import generate_data
            generate_data()
        df, metrics, agent = _build_pipeline(demo_path)
        st.session_state.df = df
        st.session_state.metrics = metrics
        st.session_state.agent = agent
        st.session_state.chat_history = []
        st.session_state.tool_traces = []
        logger.info("Demo dataset loaded.")
        st.success("✅ Demo data loaded!")


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("📊 BizMind AI")
    st.caption("Business intelligence for every shop, big or small")

    st.session_state.language = st.radio("Language / भाषा", ["English", "Hindi"])
    st.divider()

    uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    if uploaded_file and st.session_state.df is None:
        process_uploaded_file(uploaded_file)

    if st.session_state.df is None:
        if st.button("🚀 Try Demo Dataset", use_container_width=True):
            load_demo()

    st.divider()
    with st.expander("How to use"):
        st.write("1. Upload your daily sales CSV **or** click Try Demo.")
        st.write("2. Explore charts on the **Dashboard** tab.")
        st.write("3. Ask questions in the **AI Chat** tab.")
        st.write("4. Check alerts under **Smart Alerts**.")
        st.write("5. Download a PDF from **Weekly Report**.")

    if st.session_state.df is not None:
        st.divider()
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------

if st.session_state.df is None:
    st.info("👆 Upload a CSV file from the sidebar, or click **Try Demo Dataset** to explore.")
    st.stop()

df = st.session_state.df
metrics = st.session_state.metrics

tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "💬 AI Chat", "🚨 Smart Alerts", "📄 Weekly Report"])


# ==================================================================
# TAB 1: DASHBOARD
# ==================================================================
with tab1:
    st.header("Business Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{metrics['total_revenue']:,.2f}")
    c2.metric("Total Orders", f"{metrics['total_orders']:,}")
    c3.metric("Unique Customers", f"{metrics['unique_customers']:,}")
    c4.metric("Top Product", metrics["top_product"])

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Revenue by Day of Week")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_rev = df.groupby("day_of_week")["revenue"].sum().reset_index()
        day_rev["day_of_week"] = pd.Categorical(day_rev["day_of_week"], categories=day_order, ordered=True)
        day_rev = day_rev.sort_values("day_of_week")
        fig1 = px.bar(day_rev, x="day_of_week", y="revenue", color="revenue",
                      color_continuous_scale="Blues", labels={"revenue": "Revenue (₹)", "day_of_week": ""})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Top 10 Products")
        top_prods = (df.groupby("product")["revenue"].sum()
                     .reset_index().sort_values("revenue", ascending=True).tail(10))
        fig2 = px.bar(top_prods, x="revenue", y="product", orientation="h",
                      color="revenue", color_continuous_scale="Greens",
                      labels={"revenue": "Revenue (₹)", "product": ""})
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.subheader("Revenue Trend")
        trend_df = df.groupby("date")["revenue"].sum().reset_index()
        fig3 = px.line(trend_df, x="date", y="revenue",
                       labels={"revenue": "Revenue (₹)", "date": "Date"})
        fig3.update_traces(line_color="#6366f1")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    # Payment mode breakdown — India-specific insight
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Payment Mode Split")
        pay = df.groupby("payment_mode")["revenue"].sum().reset_index()
        fig4 = px.pie(pay, names="payment_mode", values="revenue",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.subheader("Category Performance")
        cat = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        fig5 = px.bar(cat, x="category", y="revenue", color="category",
                      labels={"revenue": "Revenue (₹)", "category": ""},
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig5, use_container_width=True)


# ==================================================================
# TAB 2: AI CHAT
# ==================================================================
with tab2:
    st.header("Chat with BizMind AI")

    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show tool trace for assistant messages that have one
            if msg["role"] == "assistant" and i < len(st.session_state.tool_traces):
                trace = st.session_state.tool_traces[i]
                if trace:
                    with st.expander(f"🔍 Agent used {len(trace)} tool(s) — click to inspect"):
                        for step in trace:
                            st.markdown(f"**Tool:** `{step['tool']}`")
                            st.markdown(f"**Output:** {step['output']}")
                            st.divider()

    user_query = st.chat_input("Ask about your business…")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                prompt = user_query
                if st.session_state.language == "Hindi" and "hindi" not in user_query.lower():
                    prompt = f"{user_query} (Please answer in Hindi)"

                if st.session_state.agent is None:
                    response = "⚠️ Agent not initialised. Please reload the page and re-upload your CSV."
                    trace = []
                else:
                    response, trace = st.session_state.agent.run_query(
                        prompt, thread_id=st.session_state.session_id
                    )

            st.markdown(response)
            if trace:
                with st.expander(f"🔍 Agent used {len(trace)} tool(s) — click to inspect"):
                    for step in trace:
                        st.markdown(f"**Tool:** `{step['tool']}`")
                        st.markdown(f"**Output:** {step['output']}")
                        st.divider()

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.tool_traces.append(trace)


# ==================================================================
# TAB 3: SMART ALERTS
# ==================================================================
with tab3:
    st.header("Smart Alerts")

    st.subheader("🔬 Anomaly Detection")
    with st.spinner("Running anomaly detection…"):
        set_anom(df)
        anomalies = detect_anomalies.invoke({"product_name": "all"})

    if "No significant" in anomalies or "Error" in anomalies:
        st.info(anomalies)
    else:
        for line in anomalies.splitlines():
            if not line.strip():
                continue
            if "HIGH" in line:
                st.error(f"🔴 {line}")
            elif "MEDIUM" in line:
                st.warning(f"🟡 {line}")
            else:
                st.success(f"🟢 {line}")

    st.divider()

    st.subheader("📦 Restock Predictions")
    with st.spinner("Running restock predictor…"):
        set_rest(df)
        restock_info = predict_restock.invoke({"product_name": "all"})

    if "No restock urgency" in restock_info or "Error" in restock_info:
        st.info(restock_info)
    else:
        for line in restock_info.splitlines():
            if not line.strip():
                continue
            try:
                days_str = line.split("~")[1].split("d")[0]
                days = int(days_str)
            except (IndexError, ValueError):
                days = 30
            pct = min(1.0, days / 30.0)
            st.write(line)
            st.progress(pct)


# ==================================================================
# TAB 4: WEEKLY REPORT
# ==================================================================
with tab4:
    st.header("Weekly Report Generation")

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Compiling insights and generating PDF…"):
            set_rev(df)
            forecast = forecast_revenue.invoke({"timeframe": "next week"})
            anom = detect_anomalies.invoke({"product_name": "all"})
            restk = predict_restock.invoke({"product_name": "all"})

            insights_text = (
                f"Next Week Forecast:\n{forecast}\n\n"
                f"Anomalies:\n{anom}\n\n"
                f"Restock Alerts:\n{restk}"
            )
            pdf_path = generate_weekly_report(df, insights_text)
            logger.info("PDF report generated: %s", pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
