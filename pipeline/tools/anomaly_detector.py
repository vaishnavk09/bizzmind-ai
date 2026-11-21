from langchain.tools import tool
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.tools.base import ToolContext

_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df


def _run_detection(df: pd.DataFrame, product_name: str) -> str:
    df = df.copy()
    if product_name.lower() != "all":
        df = df[df["product"].str.lower() == product_name.lower()]
        if df.empty:
            return f"No data found for product: {product_name}."

    weekly_sales = (
        df.groupby([pd.Grouper(key="date", freq="W-MON"), "product"])["quantity_sold"]
        .sum()
        .reset_index()
    )

    anomalies = []
    for product in weekly_sales["product"].unique():
        prod_data = weekly_sales[weekly_sales["product"] == product].sort_values("date")
        if len(prod_data) < 2:
            continue

        prod_data = prod_data.copy()
        prod_data["mean_4w"] = prod_data["quantity_sold"].shift(1).rolling(window=4, min_periods=1).mean()
        prod_data["std_4w"] = prod_data["quantity_sold"].shift(1).rolling(window=4, min_periods=1).std().fillna(0)

        current = prod_data.iloc[-1]
        mean, std, qty = current["mean_4w"], current["std_4w"], current["quantity_sold"]

        if pd.isna(mean) or mean == 0:
            continue

        z = (qty - mean) / std if std > 0 else 0
        pct = ((qty - mean) / mean) * 100

        if abs(z) > 2 or abs(pct) > 40:
            severity = "HIGH" if abs(pct) > 60 else ("MEDIUM" if abs(pct) > 40 else "LOW")
            direction = "dropped" if pct < 0 else "spiked"
            anomalies.append(
                f"ANOMALY [{severity}]: {product} sales {direction} {abs(pct):.0f}% "
                f"vs 4-week avg (z={z:.1f})."
            )

    return "\n".join(anomalies) if anomalies else "No significant anomalies detected."


def make_anomaly_tool(ctx: "ToolContext"):
    """Returns a detect_anomalies tool bound to the given ToolContext."""

    @tool
    def detect_anomalies(product_name: str) -> str:
        """
        Detects unusual sales spikes or drops for a product.
        Uses z-score and percentage-change over a rolling 4-week window.
        Input: product name or 'all'.
        """
        if ctx.df is None or ctx.df.empty:
            return "Error: No data available."
        return _run_detection(ctx.df, product_name)

    return detect_anomalies


@tool
def detect_anomalies(product_name: str) -> str:
    """
    Detects unusual sales spikes or drops for a product.
    Uses z-score and percentage-change over a rolling 4-week window.
    Input: product name or 'all'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available."
    return _run_detection(_CONTEXT_DF, product_name)
