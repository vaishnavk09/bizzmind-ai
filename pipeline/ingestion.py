import pandas as pd

class DataIngestionPipeline:
    """
    Handles loading, cleaning, and preprocessing of sales CSV data.
    """
    def __init__(self):
        pass

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Loads and validates CSV.
        """
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles nulls, type conversions, and date parsing.
        """
        df = df.dropna() # Basic null handling
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        if 'quantity_sold' in df.columns:
            df['quantity_sold'] = pd.to_numeric(df['quantity_sold'], errors='coerce').fillna(0)
        if 'unit_price' in df.columns:
            df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce').fillna(0)
            
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds day_of_week, week_number, month_name, is_weekend, and revenue.
        """
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.day_name()
            df['week_number'] = df['date'].dt.isocalendar().week
            df['month_name'] = df['date'].dt.month_name()
            df['is_weekend'] = df['date'].dt.dayofweek >= 5

        if 'revenue' not in df.columns and 'quantity_sold' in df.columns and 'unit_price' in df.columns:
            df['revenue'] = df['quantity_sold'] * df['unit_price']
            
        return df

    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Returns summary metrics: total_revenue, total_orders, unique_customers, 
        unique_products, date_range, top_product, top_category.
        """
        total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
        total_orders = len(df)
        unique_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
        unique_products = df['product'].nunique() if 'product' in df.columns else 0
        
        date_range = "Unknown"
        if 'date' in df.columns and not df['date'].empty:
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            
        top_product = df.groupby('product')['revenue'].sum().idxmax() if 'product' in df.columns and 'revenue' in df.columns else "None"
        top_category = df.groupby('category')['revenue'].sum().idxmax() if 'category' in df.columns and 'revenue' in df.columns else "None"

        return {
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "unique_customers": unique_customers,
            "unique_products": unique_products,
            "date_range": date_range,
            "top_product": top_product,
            "top_category": top_category
        }

    def to_text_chunks(self, df: pd.DataFrame) -> list:
        """
        Converts rows to text chunks for FAISS indexing.
        Format: 'On {date}, {customer_id} bought {qty} units of {product} ({category}) for ₹{revenue} via {payment_mode}'
        """
        chunks = []
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notnull(row.get('date')) else 'Unknown date'
            chunk = (
                f"On {date_str}, {row.get('customer_id', 'Unknown')} bought "
                f"{row.get('quantity_sold', 0)} units of {row.get('product', 'Unknown')} "
                f"({row.get('category', 'Unknown')}) for ₹{row.get('revenue', 0)} "
                f"via {row.get('payment_mode', 'Unknown')}"
            )
            chunks.append(chunk)
        return chunks

if __name__ == "__main__":
    # Test block
    pipeline = DataIngestionPipeline()
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_store.csv")
    if os.path.exists(csv_path):
        df = pipeline.load_csv(csv_path)
        df = pipeline.clean_data(df)
        df = pipeline.add_features(df)
        stats = pipeline.get_summary_stats(df)
        print("Summary Stats:", stats)
        chunks = pipeline.to_text_chunks(df)
        print("First 3 chunks:")
        for c in chunks[:3]:
            print(c)
    else:
        print("CSV not found. Generate it first.")
