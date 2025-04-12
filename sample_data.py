import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_sales_data(rows=1000, output_file="sample_sales_data.xlsx"):
    """
    Generate a sample sales dataset and save it as an Excel file
    
    Args:
        rows: Number of rows to generate
        output_file: Path to save the Excel file
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # Product categories and products
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports']
    products = {
        'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
        'Clothing': ['T-shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
        'Home & Kitchen': ['Blender', 'Coffee Maker', 'Toaster', 'Cookware Set', 'Vacuum Cleaner'],
        'Books': ['Fiction', 'Non-fiction', 'Biography', 'Textbook', 'Children\'s Book'],
        'Sports': ['Running Shoes', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Basketball']
    }
    
    # Regions and cities
    regions = {
        'North': ['New York', 'Boston', 'Chicago', 'Detroit'],
        'South': ['Miami', 'Atlanta', 'Dallas', 'Houston'],
        'West': ['Los Angeles', 'San Francisco', 'Seattle', 'Denver'],
        'East': ['Philadelphia', 'Washington DC', 'Baltimore', 'Pittsburgh']
    }
    
    # Customer segments
    segments = ['New', 'Returning', 'Loyal', 'VIP']
    
    # Payment methods
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']
    
    # Generate data
    data = {
        'OrderID': [f'ORD-{i:06d}' for i in range(1, rows + 1)],
        'Date': [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(rows)],
        'CustomerID': [f'CUST-{random.randint(1, 500):04d}' for _ in range(rows)],
        'CustomerSegment': [random.choice(segments) for _ in range(rows)],
        'Region': [random.choice(list(regions.keys())) for _ in range(rows)],
        'PaymentMethod': [random.choice(payment_methods) for _ in range(rows)],
        'Discount': [round(random.uniform(0, 0.3), 2) for _ in range(rows)],
        'ShippingCost': [round(random.uniform(5, 20), 2) for _ in range(rows)]
    }
    
    # Add cities based on regions
    data['City'] = [random.choice(regions[region]) for region in data['Region']]
    
    # Add categories and products
    data['Category'] = [random.choice(categories) for _ in range(rows)]
    data['Product'] = [random.choice(products[category]) for category in data['Category']]
    
    # Add quantities and prices
    data['Quantity'] = [random.randint(1, 5) for _ in range(rows)]
    
    # Set price ranges by category
    price_ranges = {
        'Electronics': (500, 2000),
        'Clothing': (20, 200),
        'Home & Kitchen': (50, 500),
        'Books': (10, 50),
        'Sports': (30, 300)
    }
    
    data['UnitPrice'] = [round(random.uniform(*price_ranges[category]), 2) for category in data['Category']]
    
    # Calculate total sales
    data['TotalSales'] = [round(qty * price * (1 - discount), 2) 
                         for qty, price, discount in zip(data['Quantity'], data['UnitPrice'], data['Discount'])]
    
    # Add some random ratings
    data['CustomerRating'] = [random.randint(1, 5) if random.random() > 0.2 else None for _ in range(rows)]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values
    for col in ['CustomerSegment', 'ShippingCost', 'Discount']:
        mask = np.random.random(size=len(df)) < 0.05
        df.loc[mask, col] = None
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Sample data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    generate_sample_sales_data(output_file="sample_sales_data.xlsx")
