import streamlit as st
import pandas as pd
import os
import plotly.express as px
from pipeline.ingestion import DataIngestionPipeline
from pipeline.vector_store import VectorStoreManager
from pipeline.agent import BizMindAgent
from pipeline.report import generate_weekly_report
from pipeline.tools.anomaly_detector import detect_anomalies
from pipeline.tools.restock_predictor import predict_restock
from pipeline.tools.revenue_forecaster import forecast_revenue
from dotenv import load_dotenv

# Page config
st.set_page_config(
    page_title="BizMind AI",
    page_icon="🧠",
    layout="wide"
)

# Load env variables
load_dotenv()

# Initialize session state variables
if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "language" not in st.session_state:
    st.session_state.language = "English"

def process_uploaded_file(file):
    with st.spinner("Processing data..."):
        # Save temp file
        temp_path = "temp_uploaded.csv"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
            
        pipeline = DataIngestionPipeline()
        df = pipeline.load_csv(temp_path)
        df = pipeline.clean_data(df)
        df = pipeline.add_features(df)
        
        st.session_state.df = df
        st.session_state.metrics = pipeline.get_summary_stats(df)
        
        # Build Vector Store
        chunks = pipeline.to_text_chunks(df)
        vsm = VectorStoreManager(index_path="faiss_index")
        vsm.build_store(chunks)
        
        # Initialize Agent
        st.session_state.agent = BizMindAgent(df, vsm)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        st.success("Data processed successfully!")

# Sidebar
with st.sidebar:
    st.title("🧠 BizMind AI")
    st.caption("BI for every business, big or small")
    
    st.session_state.language = st.radio("Language / भाषा", ["English", "Hindi"])
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    if uploaded_file is not None and st.session_state.df is None:
        process_uploaded_file(uploaded_file)
        
    st.divider()
    
    with st.expander("How to use"):
        st.write("1. Upload your daily sales CSV file.")
        st.write("2. View automated charts on the Dashboard.")
        st.write("3. Ask BizMind AI questions in the AI Chat tab.")
        
# Main Area
if st.session_state.df is not None:
    df = st.session_state.df
    metrics = st.session_state.metrics
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "AI Chat", "Smart Alerts", "Weekly Report"])
    
    # --- TAB 1: Dashboard ---
    with tab1:
        st.header("Business Overview")
        
        # 4 Columns for Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"₹{metrics['total_revenue']:,.2f}")
        col2.metric("Total Orders", metrics['total_orders'])
        col3.metric("Unique Customers", metrics['unique_customers'])
        col4.metric("Top Product", metrics['top_product'])
        
        st.divider()
        
        # 3 Plotly Charts
        col_chart1, col_chart2, col_chart3 = st.columns(3)
        
        with col_chart1:
            st.subheader("Revenue by Day of Week")
            day_rev = df.groupby('day_of_week')['revenue'].sum().reset_index()
            # Order days logically
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_rev['day_of_week'] = pd.Categorical(day_rev['day_of_week'], categories=days, ordered=True)
            day_rev = day_rev.sort_values('day_of_week')
            fig1 = px.bar(day_rev, x='day_of_week', y='revenue', color='revenue', color_continuous_scale='Blues')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            st.subheader("Top 10 Products")
            top_prods = df.groupby('product')['revenue'].sum().reset_index().sort_values('revenue', ascending=True).tail(10)
            fig2 = px.bar(top_prods, x='revenue', y='product', orientation='h', color='revenue', color_continuous_scale='Greens')
            st.plotly_chart(fig2, use_container_width=True)
            
        with col_chart3:
            st.subheader("Revenue Trend")
            trend_df = df.groupby('date')['revenue'].sum().reset_index()
            fig3 = px.line(trend_df, x='date', y='revenue')
            st.plotly_chart(fig3, use_container_width=True)

    # --- TAB 2: AI Chat ---
    with tab2:
        st.header("Chat with BizMind AI")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat input
        user_query = st.chat_input("Ask about your business...")
        if user_query:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    if st.session_state.language == "Hindi" and "hindi" not in user_query.lower():
                        prompt = f"{user_query} (Please answer in Hindi)"
                    else:
                        prompt = user_query
                        
                    if st.session_state.agent is None:
                        response = "⚠️ The AI Agent wasn't initialized properly (likely due to a previous error). Please refresh the page and re-upload your CSV file."
                    else:
                        response = st.session_state.agent.run_query(prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # --- TAB 3: Smart Alerts ---
    with tab3:
        st.header("Smart Alerts")
        
        st.subheader("Anomaly Detection")
        with st.spinner("Running anomaly detection..."):
            from pipeline.tools.anomaly_detector import set_context_df as set_anom
            set_anom(df)
            anomalies = detect_anomalies.invoke({"product_name": "all"})
            
            if "No significant" in anomalies or "Error" in anomalies:
                st.info(anomalies)
            else:
                for line in anomalies.split('\n'):
                    if not line.strip(): continue
                    if "HIGH" in line:
                        st.error(f"🔴 {line}")
                    elif "MEDIUM" in line:
                        st.warning(f"🟡 {line}")
                    else:
                        st.success(f"🟢 {line}")
                        
        st.divider()
        
        st.subheader("Restock Predictions")
        with st.spinner("Running restock predictor..."):
            from pipeline.tools.restock_predictor import set_context_df as set_rest
            set_rest(df)
            restock_info = predict_restock.invoke({"product_name": "all"})
            
            if "No restock urgency" in restock_info or "Error" in restock_info:
                st.info(restock_info)
            else:
                for line in restock_info.split('\n'):
                    if not line.strip(): continue
                    product_part = line.split(':')[0]
                    days_part = line.split('~')[1].split(' ')[0] if '~' in line else 30
                    try:
                        days = int(days_part)
                    except:
                        days = 30
                        
                    pct = min(1.0, days / 30.0)
                    
                    st.write(line)
                    st.progress(pct)

    # --- TAB 4: Weekly Report ---
    with tab4:
        st.header("Weekly Report Generation")
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Compiling insights and generating PDF..."):
                # Compile some insights
                from pipeline.tools.revenue_forecaster import set_context_df as set_rev
                set_rev(df)
                forecast = forecast_revenue.invoke({"timeframe": "next week"})
                anom = detect_anomalies.invoke({"product_name": "all"})
                restk = predict_restock.invoke({"product_name": "all"})
                
                insights_text = f"Next Week Forecast:\n{forecast}\n\nAnomalies:\n{anom}\n\nRestock Alerts:\n{restk}"
                
                pdf_path = generate_weekly_report(df, insights_text)
                
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_file,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
else:
    st.info("Please upload a CSV file from the sidebar to begin.")
