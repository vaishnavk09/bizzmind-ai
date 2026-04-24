from langchain.tools import tool
import pandas as pd
import numpy as np

_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df

@tool
def forecast_revenue(timeframe: str) -> str:
    """
    Predicts next week or next month revenue using linear regression.
    Input should be 'next week' or 'next month'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available."

    df = _CONTEXT_DF.copy()
    
    if "week" in timeframe.lower():
        freq = 'W-MON'
        period_name = "week"
        trend_periods = 4
    elif "month" in timeframe.lower():
        freq = 'ME' # pandas 2.2 uses 'ME' for month end
        period_name = "month"
        trend_periods = 3 # last 3 months
    else:
        return "Please specify 'next week' or 'next month'."

    # Group by freq
    revenue_data = df.groupby(pd.Grouper(key='date', freq=freq))['revenue'].sum().reset_index()
    
    if len(revenue_data) < trend_periods:
        return "Not enough data to generate forecast."
        
    # Use last N periods
    recent_data = revenue_data.tail(trend_periods).copy()
    
    # Linear regression using numpy polyfit
    x = np.arange(len(recent_data))
    y = recent_data['revenue'].values
    
    # fit a degree-1 polynomial (line)
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Predict next period (x = len(recent_data))
    next_x = len(recent_data)
    forecast = (slope * next_x) + intercept
    
    current_period_rev = y[-1]
    
    if current_period_rev > 0:
        pct_change = ((forecast - current_period_rev) / current_period_rev) * 100
        sign = "+" if pct_change > 0 else ""
        return f"Projected next {period_name} revenue: ₹{max(0, forecast):,.2f} ({sign}{pct_change:.1f}% vs this {period_name})."
    else:
        return f"Projected next {period_name} revenue: ₹{max(0, forecast):,.2f}."
