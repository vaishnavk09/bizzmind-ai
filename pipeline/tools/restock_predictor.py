from langchain.tools import tool
import pandas as pd
import numpy as np

_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df

@tool
def predict_restock(product_name: str) -> str:
    """
    Estimates days until stockout for a product.
    Input should be a product name or 'all'.
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available."

    df = _CONTEXT_DF.copy()
    
    if product_name.lower() != "all":
        df = df[df['product'].str.lower() == product_name.lower()]
        if df.empty:
            return f"No data found for product: {product_name}."

    predictions = []
    
    for product in df['product'].unique():
        prod_data = df[df['product'] == product]
        
        # Calculate total days in dataset
        days_active = (prod_data['date'].max() - prod_data['date'].min()).days
        if days_active <= 0:
            days_active = 1
            
        avg_daily_sales = prod_data['quantity_sold'].sum() / days_active
        
        # Max weekly sales * 2 as assumed starting stock
        weekly_sales = prod_data.groupby(pd.Grouper(key='date', freq='W'))['quantity_sold'].sum()
        max_weekly_sales = weekly_sales.max()
        
        # Simplified logic for remaining stock
        # Assume current stock = (Max weekly sales * 2) - sales in last 7 days
        recent_sales = prod_data[prod_data['date'] >= (prod_data['date'].max() - pd.Timedelta(days=7))]['quantity_sold'].sum()
        estimated_stock = (max_weekly_sales * 2) - recent_sales
        
        if avg_daily_sales > 0:
            days_remaining = estimated_stock / avg_daily_sales
        else:
            days_remaining = 999
            
        if days_remaining <= 7:
            predictions.append(f"{product}: ~{max(0, int(days_remaining))} days remaining. RESTOCK NOW.")
        elif days_remaining <= 14:
            predictions.append(f"{product}: ~{int(days_remaining)} days remaining. Order within a week.")
        else:
            if product_name.lower() != "all":
                predictions.append(f"{product}: ~{int(days_remaining)} days remaining. Stock is healthy.")

    if not predictions:
        return "No restock urgency detected."
        
    return "\n".join(predictions)
