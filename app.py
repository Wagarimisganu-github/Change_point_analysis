import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data with error handling
try:
    df = pd.read_csv('data/processed/cleaned_oil_prices.csv', parse_dates=['Date'])
except FileNotFoundError:
    print("Error: Missing cleaned oil prices data")
    print("Please run data processing scripts first")
    exit(1)

# Try to load events data
try:
    events = pd.read_csv('data/processed/events_annotated.csv', parse_dates=['Date'])
    has_events = True
except FileNotFoundError:
    events = pd.DataFrame(columns=['Date', 'Event'])
    has_events = False
    print("Note: Running without event annotations")

# Create app layout
app.layout = html.Div([
    html.H1("Brent Oil Price Analysis Dashboard"),
    
    dcc.Graph(
        id='price-chart',
        figure=px.line(df, x='Date', y='Price', title='Historical Brent Oil Prices')
    ),
    
    html.Div([
        html.H3("Key Statistics:"),
        html.P(f"Time Period: {df['Date'].min().date()} to {df['Date'].max().date()}"),
        html.P(f"Average Price: ${df['Price'].mean():.2f}"),
        html.P(f"Maximum Price: ${df['Price'].max():.2f}"),
        html.P(f"Minimum Price: ${df['Price'].min():.2f}"),
    ])
])

if __name__ == '__main__':
    # Updated to use app.run() instead of app.run_server()
    app.run(debug=True)