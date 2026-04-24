import os
import pandas as pd
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

from pipeline.tools.trend_analyzer import analyze_trends, set_context_df as set_trend_df
from pipeline.tools.anomaly_detector import detect_anomalies, set_context_df as set_anomaly_df
from pipeline.tools.restock_predictor import predict_restock, set_context_df as set_restock_df
from pipeline.tools.revenue_forecaster import forecast_revenue, set_context_df as set_revenue_df
from pipeline.tools.insight_generator import generate_insight, init_insight_generator

load_dotenv()

class BizMindAgent:
    def __init__(self, df: pd.DataFrame, vsm):
        self.df = df
        self.vsm = vsm
        
        # Initialize Groq LLM
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            print("WARNING: GROQ_API_KEY not set. Agent will not work properly.")
            
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key
        )
        
        self.memory = InMemorySaver()
        self.agent = self._setup_agent()

    def _setup_agent(self):
        # Bind dataframe to tools
        set_trend_df(self.df)
        set_anomaly_df(self.df)
        set_restock_df(self.df)
        set_revenue_df(self.df)
        
        # Init insight generator
        init_insight_generator(self.vsm, self.llm)
        
        tools = [
            analyze_trends,
            detect_anomalies,
            predict_restock,
            forecast_revenue,
            generate_insight
        ]
        
        # System prompt instructions
        system_message = (
            "You are BizMind AI, a friendly business analyst for small shop owners in India. "
            "Always answer in simple language. Use ₹ for currency. Be specific with numbers. "
            "If asked in Hindi, respond in Hindi. Always end with one actionable recommendation."
        )
        
        agent_executor = create_react_agent(
            model=self.llm, 
            tools=tools, 
            prompt=system_message,
            checkpointer=self.memory
        )
        return agent_executor

    def run_query(self, question: str) -> str:
        """
        Runs the agent on the user question and returns the answer string.
        """
        try:
            config = {"configurable": {"thread_id": "bizmind_session"}}
            response = self.agent.invoke({"messages": [("user", question)]}, config=config)
            
            messages = response.get("messages", [])
            if messages:
                return messages[-1].content
            return "No response generated."
        except Exception as e:
            return f"Error executing query: {str(e)}"
