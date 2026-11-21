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


def _bootstrap_ci(values: np.ndarray, n_boot: int = 500, ci: float = 0.90) -> tuple[float, float]:
    """Returns (lower, upper) confidence interval via bootstrap resampling."""
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def _run_forecast(df: pd.DataFrame, timeframe: str) -> str:
    if "week" in timeframe.lower():
        freq, period_name, trend_periods = "W-MON", "week", 4
    elif "month" in timeframe.lower():
        freq, period_name, trend_periods = "ME", "month", 3
    else:
        return "Please specify 'next week' or 'next month'."

    revenue_data = df.groupby(pd.Grouper(key="date", freq=freq))["revenue"].sum().reset_index()

    if len(revenue_data) < trend_periods:
        return "Not enough data to generate forecast."

    recent = revenue_data.tail(trend_periods).copy()
    x = np.arange(len(recent))
    y = recent["revenue"].values

    # Linear trend
    coeffs = np.polyfit(x, y, 1)
    forecast = float(coeffs[0] * len(recent) + coeffs[1])
    forecast = max(0.0, forecast)

    # Confidence interval via residual bootstrap
    residuals = y - np.polyval(coeffs, x)
    ci_lo, ci_hi = _bootstrap_ci(residuals)
    lo = max(0.0, forecast + ci_lo)
    hi = forecast + ci_hi

    current = float(y[-1])
    pct = ((forecast - current) / current * 100) if current > 0 else 0
    sign = "+" if pct >= 0 else ""

    return (
        f"Projected next {period_name} revenue: ₹{forecast:,.2f} "
        f"({sign}{pct:.1f}% vs this {period_name}). "
        f"90% CI: ₹{lo:,.2f} – ₹{hi:,.2f}."
    )


def make_forecast_tool(ctx: "ToolContext"):
    """Returns a forecast_revenue tool bound to the given ToolContext."""

    @tool
    def forecast_revenue(timeframe: str) -> str:
        """
        Predicts next week or next month revenue using linear regression
        with a 90% bootstrap confidence interval.
        Input: 'next week' or 'next month'.
        """
        if ctx.df is None or ctx.df.empty:
            return "Error: No data available."
        return _run_forecast(ctx.df, timeframe)

    return forecast_revenue


@tool
def forecast_revenue(timeframe: str) -> str:
    """
    Predicts next week or next month revenue using linear regression
    with a 90% bootstrap confidence interval.
    Input: 'next week' or 'next month'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available."
    return _run_forecast(_CONTEXT_DF, timeframe)
