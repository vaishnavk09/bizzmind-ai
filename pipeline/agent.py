import os
import logging
import pandas as pd
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from pipeline.tools.base import ToolContext
from pipeline.tools.trend_analyzer import make_trend_tool
from pipeline.tools.anomaly_detector import make_anomaly_tool
from pipeline.tools.restock_predictor import make_restock_tool
from pipeline.tools.revenue_forecaster import make_forecast_tool
from pipeline.tools.insight_generator import make_insight_tool
from pipeline.vector_store import VectorStoreManager

load_dotenv()
logger = logging.getLogger(__name__)

# Few-shot examples embedded in the system prompt so the agent learns
# the expected answer style without extra LLM calls.
_SYSTEM_PROMPT = """\
You are BizMind AI, a sharp business analyst for small shop owners in India.

## Persona
- Speak plainly. No jargon. Use ₹ for currency.
- Be specific: always include numbers, percentages, or dates from the data.
- End every answer with one clear, actionable recommendation prefixed with "→ Action:".
- If asked in Hindi, reply fully in Hindi.

## Tool use rules
- Use `analyze_trends` for product, day, category, or payment-mode rankings.
- Use `detect_anomalies` when the user asks about unusual changes or spikes/drops.
- Use `predict_restock` when the user asks about inventory, stockouts, or when to reorder.
- Use `forecast_revenue` when the user asks about future revenue or projections.
- Use `generate_insight` for open-ended questions that need context from past sales records.
- Never fabricate numbers. If a tool returns no data, say so clearly.

## Few-shot examples

User: Which product sells the most?
Thought: This is a ranking question → use analyze_trends.
Answer: Rice is the top seller at ₹1,24,500 revenue (18% of total).
→ Action: Ensure Rice stock never falls below 2 weeks of supply.

User: Should I restock Sugar?
Thought: Stockout question → use predict_restock.
Answer: Sugar has ~5 days of stock remaining at current sales pace.
→ Action: Order Sugar today — delivery lead time likely exceeds remaining stock.

User: What is unusual about this week?
Thought: Anomaly detection question → use detect_anomalies.
Answer: Shampoo sales spiked 72% above the 4-week average this week (z=2.4, HIGH severity).
→ Action: Check if a promotion drove this; if not, investigate supply to meet continued demand.
"""


class BizMindAgent:
    def __init__(self, df: pd.DataFrame, vsm: VectorStoreManager):
        self.df = df
        self.vsm = vsm

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY not set — agent will not work.")

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key,
        )

        self.memory = InMemorySaver()
        self._tool_calls: list[dict] = []   # accumulates trace per session
        self.agent = self._setup_agent()

    def _setup_agent(self):
        ctx = ToolContext(df=self.df, vsm=self.vsm)

        tools = [
            make_trend_tool(ctx),
            make_anomaly_tool(ctx),
            make_restock_tool(ctx),
            make_forecast_tool(ctx),
            make_insight_tool(ctx, self.llm),
        ]

        agent_executor = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=_SYSTEM_PROMPT,
            checkpointer=self.memory,
        )
        logger.info("BizMindAgent initialized with %d tools.", len(tools))
        return agent_executor

    def run_query(self, question: str, thread_id: str = "default") -> tuple[str, list[dict]]:
        """
        Runs the agent on the user question.
        Returns (answer_text, tool_trace) where tool_trace is a list of
        {tool, input, output} dicts for the UI trace panel.
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            response = self.agent.invoke({"messages": [("user", question)]}, config=config)

            messages = response.get("messages", [])
            tool_trace = []

            for msg in messages:
                # LangGraph emits ToolMessage for each tool call result
                if hasattr(msg, "name") and hasattr(msg, "content") and msg.type == "tool":
                    tool_trace.append({"tool": msg.name, "output": msg.content})

            answer = messages[-1].content if messages else "No response generated."
            logger.info("Query answered. Tools used: %s", [t["tool"] for t in tool_trace])
            return answer, tool_trace

        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            return f"Error executing query: {str(e)}", []
