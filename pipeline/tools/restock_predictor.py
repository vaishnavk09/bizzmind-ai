from langchain.tools import tool
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.tools.base import ToolContext

_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df


def _run_prediction(df: pd.DataFrame, product_name: str) -> str:
    df = df.copy()
    if product_name.lower() != "all":
        df = df[df["product"].str.lower() == product_name.lower()]
        if df.empty:
            return f"No data found for product: {product_name}."

    predictions = []
    for product in df["product"].unique():
        prod = df[df["product"] == product]

        days_active = (prod["date"].max() - prod["date"].min()).days or 1
        avg_daily = prod["quantity_sold"].sum() / days_active

        weekly = prod.groupby(pd.Grouper(key="date", freq="W"))["quantity_sold"].sum()
        max_weekly = weekly.max()

        cutoff = prod["date"].max() - pd.Timedelta(days=7)
        recent = prod[prod["date"] >= cutoff]["quantity_sold"].sum()
        estimated_stock = max(0, (max_weekly * 2) - recent)

        days_left = int(estimated_stock / avg_daily) if avg_daily > 0 else 999

        if days_left <= 7:
            predictions.append(f"{product}: ~{days_left}d left. ⚠️  RESTOCK NOW.")
        elif days_left <= 14:
            predictions.append(f"{product}: ~{days_left}d left. Order within a week.")
        elif product_name.lower() != "all":
            predictions.append(f"{product}: ~{days_left}d left. Stock is healthy.")

    return "\n".join(predictions) if predictions else "No restock urgency detected."


def make_restock_tool(ctx: "ToolContext"):
    """Returns a predict_restock tool bound to the given ToolContext."""

    @tool
    def predict_restock(product_name: str) -> str:
        """
        Estimates days until stockout for a product based on average daily sales velocity.
        Input: product name or 'all'.
        """
        if ctx.df is None or ctx.df.empty:
            return "Error: No data available."
        return _run_prediction(ctx.df, product_name)

    return predict_restock


@tool
def predict_restock(product_name: str) -> str:
    """
    Estimates days until stockout for a product based on average daily sales velocity.
    Input: product name or 'all'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available."
    return _run_prediction(_CONTEXT_DF, product_name)
