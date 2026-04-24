import pandas as pd
import numpy as np
import os

def generate_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=180)
    products = ["Rice", "Cooking Oil", "Sugar", "Dal", "Soap",
                "Biscuits", "Tea", "Salt", "Shampoo", "Chips"]
    customers = [f"CUST{str(i).zfill(3)}" for i in range(1, 81)]

    data = pd.DataFrame({
        "date": np.random.choice(dates, 600),
        "product": np.random.choice(products, 600),
        "quantity_sold": np.random.randint(1, 50, 600),
        "unit_price": np.random.randint(20, 500, 600),
        "customer_id": np.random.choice(customers, 600),
        "category": np.random.choice(
            ["Grocery", "Personal Care", "Snacks"], 600
        ),
        "payment_mode": np.random.choice(
            ["Cash", "UPI", "Credit"], 600, p=[0.5, 0.4, 0.1]
        )
    })
    data["revenue"] = data["quantity_sold"] * data["unit_price"]
    data["date"] = pd.to_datetime(data["date"])
    
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), "mock_store.csv")
    data.to_csv(csv_path, index=False)
    print(f"Mock data generated successfully at {csv_path}")

if __name__ == "__main__":
    generate_data()
