import pandas as pd
import numpy as np
import os

def load_and_preprocess():
    """Load and preprocess Brent oil price data."""
    # Load raw data with explicit date parsing
    df = pd.read_csv(
        './data/raw/brent_oil_prices.csv',
        parse_dates=['Date'],
        dayfirst=True  # Important for day-month-year dates
    )
    
    # Clean price data - handle both string and numeric cases
    if not pd.api.types.is_numeric_dtype(df['Price']):
        # If Price is string, remove commas and convert
        df['Price'] = pd.to_numeric(
            df['Price'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
    else:
        # If already numeric, just ensure float type
        df['Price'] = df['Price'].astype(float)
    
    # Drop any rows with missing prices
    df = df.dropna(subset=['Price'])
    
    # Add derived features
    df['Log_Price'] = np.log(df['Price'])
    df['Returns'] = df['Price'].pct_change()
    
    # Save processed data
    os.makedirs('./data/processed', exist_ok=True)
    df.to_csv('./data/processed/cleaned_oil_prices.csv', index=False)
    
    return df

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess()
    print(f"✅ Success! Processed {len(df)} records")
    print("Sample data:")
    print(df.head())