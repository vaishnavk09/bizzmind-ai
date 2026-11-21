from langchain.tools import tool
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.tools.base import ToolContext

# Legacy global kept for backward-compat; use make_trend_tool() in new code.
_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df


def _run_analysis(df: pd.DataFrame, query: str) -> str:
    query = query.lower()
    try:
        if "product" in query:
            top = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(3)
            items = [f"{p} (₹{r:,.2f})" for p, r in top.items()]
            return "Top 3 products by revenue: " + ", ".join(items) + "."

        elif "day" in query:
            peak = df.groupby("day_of_week")["revenue"].sum().sort_values(ascending=False)
            return f"Peak day: {peak.index[0]} with ₹{peak.iloc[0]:,.2f} total revenue."

        elif "category" in query:
            cats = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
            total = df["revenue"].sum()
            pct = (cats.iloc[0] / total * 100) if total > 0 else 0
            return f"Best category: {cats.index[0]} ({pct:.1f}% of total sales)."

        elif "payment" in query or "mode" in query:
            modes = df.groupby("payment_mode")["revenue"].sum().sort_values(ascending=False)
            total = df["revenue"].sum()
            lines = [f"{m}: ₹{r:,.2f} ({r/total*100:.1f}%)" for m, r in modes.items()]
            return "Payment breakdown — " + ", ".join(lines) + "."

        else:
            top_prod = df.groupby("product")["revenue"].sum().idxmax()
            top_rev = df.groupby("product")["revenue"].sum().max()
            best_day = df.groupby("day_of_week")["revenue"].sum().idxmax()
            best_cat = df.groupby("category")["revenue"].sum().idxmax()
            total = df["revenue"].sum()
            pct = df.groupby("category")["revenue"].sum().max() / total * 100 if total > 0 else 0
            return (
                f"Top product: {top_prod} (₹{top_rev:,.2f}). "
                f"Peak day: {best_day}. "
                f"Best category: {best_cat} ({pct:.0f}% of sales)."
            )
    except Exception as e:
        return f"Error analyzing trends: {e}"


def make_trend_tool(ctx: "ToolContext"):
    """Returns an analyze_trends tool bound to the given ToolContext."""

    @tool
    def analyze_trends(query: str) -> str:
        """
        Useful for answering questions about top products, peak days, best categories,
        or payment mode breakdowns.
        Input: a string like 'top products', 'peak days', 'best category', 'payment modes'.
        """
        if ctx.df is None or ctx.df.empty:
            return "Error: No data available for analysis."
        return _run_analysis(ctx.df, query)

    return analyze_trends


# Legacy @tool used when tools are still invoked without factory (Smart Alerts tab, etc.)
@tool
def analyze_trends(query: str) -> str:
    """
    Useful for answering questions about top products, peak days, best categories,
    or payment mode breakdowns.
    Input: a string like 'top products', 'peak days', 'best category', 'payment modes'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available for analysis."
    return _run_analysis(_CONTEXT_DF, query)
