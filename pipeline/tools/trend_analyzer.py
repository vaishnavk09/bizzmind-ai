from langchain.tools import tool
import pandas as pd

# Global state to hold the dataframe for the tools
# Alternatively, this could be passed via tool kwargs, but LangChain tools
# with fixed schemas are easier if they just access a bound state or global.
# We'll set this from agent.py during initialization.
_CONTEXT_DF = None

def set_context_df(df: pd.DataFrame):
    global _CONTEXT_DF
    _CONTEXT_DF = df

@tool
def analyze_trends(query: str) -> str:
    """
    Useful for answering questions about top products, peak days, or best categories.
    Input should be a string indicating what to analyze (e.g., 'top products', 'peak days', 'best category').
    """
    if _CONTEXT_DF is None or _CONTEXT_DF.empty:
        return "Error: No data available for analysis."

    df = _CONTEXT_DF
    query = query.lower()
    
    try:
        if "product" in query:
            top_products = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(3)
            result = "Top 3 products by revenue: "
            items = []
            for prod, rev in top_products.items():
                items.append(f"{prod} (₹{rev:,.2f})")
            return result + ", ".join(items) + "."
            
        elif "day" in query:
            peak_days = df.groupby('day_of_week')['revenue'].sum().sort_values(ascending=False)
            best_day = peak_days.index[0]
            best_rev = peak_days.iloc[0]
            return f"Peak day: {best_day} with ₹{best_rev:,.2f} total revenue."
            
        elif "category" in query:
            categories = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
            best_cat = categories.index[0]
            best_rev = categories.iloc[0]
            total_rev = df['revenue'].sum()
            pct = (best_rev / total_rev) * 100 if total_rev > 0 else 0
            return f"Best category: {best_cat} ({pct:.1f}% of total sales)."
            
        else:
            # General summary
            top_prod = df.groupby('product')['revenue'].sum().idxmax()
            top_prod_rev = df.groupby('product')['revenue'].sum().max()
            best_day = df.groupby('day_of_week')['revenue'].sum().idxmax()
            best_cat = df.groupby('category')['revenue'].sum().idxmax()
            total_rev = df['revenue'].sum()
            pct = (df.groupby('category')['revenue'].sum().max() / total_rev) * 100 if total_rev > 0 else 0
            
            return f"Top product: {top_prod} (₹{top_prod_rev:,.2f} revenue). Peak day: {best_day}. Best category: {best_cat} ({pct:.0f}% of sales)."
            
    except Exception as e:
        return f"Error analyzing trends: {e}"
