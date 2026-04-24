from langchain.tools import tool
import pandas as pd
import numpy as np

# Global context set by agent.py
_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df

@tool
def detect_anomalies(product_name: str) -> str:
    """
    Detects anomalies in sales (unusual drops or spikes).
    Input should be a product name or 'all'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available."

    df = _CONTEXT_DF.copy()
    
    if product_name.lower() != "all":
        df = df[df['product'].str.lower() == product_name.lower()]
        if df.empty:
            return f"No data found for product: {product_name}."

    # Group by week and product
    weekly_sales = df.groupby([pd.Grouper(key='date', freq='W-MON'), 'product'])['quantity_sold'].sum().reset_index()
    
    anomalies = []
    
    for product in weekly_sales['product'].unique():
        prod_data = weekly_sales[weekly_sales['product'] == product].sort_values('date')
        if len(prod_data) < 2:
            continue
            
        # Calculate rolling mean and std over the last 4 weeks (roughly 30 days)
        # Shift by 1 so the current week is not included in the mean
        prod_data['mean_4w'] = prod_data['quantity_sold'].shift(1).rolling(window=4, min_periods=1).mean()
        prod_data['std_4w'] = prod_data['quantity_sold'].shift(1).rolling(window=4, min_periods=1).std().fillna(0)
        
        current_week = prod_data.iloc[-1]
        mean = current_week['mean_4w']
        std = current_week['std_4w']
        current_qty = current_week['quantity_sold']
        
        if pd.isna(mean) or mean == 0:
            continue
            
        # Flag if current week > 2 std devs from mean
        # Or if it's a huge percentage drop
        z_score = (current_qty - mean) / std if std > 0 else 0
        pct_change = ((current_qty - mean) / mean) * 100
        
        if abs(z_score) > 2 or abs(pct_change) > 40:
            severity = "HIGH" if abs(pct_change) > 60 else ("MEDIUM" if abs(pct_change) > 40 else "LOW")
            direction = "dropped" if pct_change < 0 else "spiked"
            anomalies.append(f"ANOMALY [{severity}]: {product} sales {direction} {abs(pct_change):.0f}% vs last month average.")

    if not anomalies:
        return "No significant anomalies detected."
    
    return "\n".join(anomalies)
