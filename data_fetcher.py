# data_fetcher.py
import pandas as pd
from fredapi import Fred
import wbdata
import datetime
import os # Import os to access environment variables

# --- Securely get API Key ---
# Try to get the key from an environment variable first
FRED_API_KEY = os.getenv(os.getenv('FRED_API_KEY'))

# If not found in environment variables, fallback to placeholder
# IMPORTANT: For production or sharing, avoid hardcoding keys.
# Use environment variables or a secure config method.
if not FRED_API_KEY:
    FRED_API_KEY = 'YOUR_FRED_API_KEY' # Replace this ONLY if not using environment variables
    print("Warning: FRED API Key not found in environment variables. Using placeholder/hardcoded key.")

try:
    # Initialize FRED API client
    fred = Fred(api_key=FRED_API_KEY)
    if FRED_API_KEY == 'YOUR_FRED_API_KEY':
         raise ValueError("Placeholder API Key detected.") # Force error if key wasn't replaced/set
except Exception as e:
    print(f"Error initializing FRED API: {e}")
    print("Please ensure you have a valid FRED API key.")
    print("Set it as an environment variable 'FRED_API_KEY' or replace 'YOUR_FRED_API_KEY' in data_fetcher.py (less secure).")
    fred = None

# --- FRED Data Fetching ---
def get_fred_series_latest(series_id):
    """Fetches the latest value for a FRED series."""
    if not fred:
        return "FRED API not initialized. Cannot fetch data."
    try:
        data = fred.get_series(series_id)
        # Drop NaN values that might be at the end
        data = data.dropna()
        if data.empty:
            return f"No valid data found for FRED series {series_id} after removing NaNs."
        latest_value = data.iloc[-1] # Get the last valid row/value
        latest_date = data.index[-1].strftime('%Y-%m-%d') # Get the last date
        # Basic formatting (adjust based on indicator needs)
        if abs(latest_value) > 1000: # Example: Format large numbers
             return f"{latest_value:,.2f} (as of {latest_date})"
        else:
             return f"{latest_value:.2f} (as of {latest_date})"
    except Exception as e:
        # Be more specific about potential errors if possible
        if 'API key' in str(e):
             return "Error fetching FRED series: Invalid API Key. Please check your FRED_API_KEY."
        return f"Error fetching FRED series {series_id}: {e}"

# --- World Bank Data (Example - remains same for now) ---
def get_world_bank_indicator_latest(indicator_code, country_iso2='US'):
    # (Keep the existing World Bank function as is for now)
    try:
        end_year = datetime.datetime.now().year
        start_year = end_year - 20
        data = wbdata.get_dataframe(
            {indicator_code: "Value"},
            country=country_iso2,
            data_date=(datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 1, 1)),
            convert_date=True
        )
        if data.empty: return f"No data for WB {indicator_code} for {country_iso2}."
        data = data.dropna().sort_index()
        if data.empty: return f"No recent valid data for WB {indicator_code} for {country_iso2}."
        latest_value = data['Value'].iloc[-1]
        latest_date = data.index[-1].strftime('%Y')
        return f"{latest_value:.2f} (as of {latest_date})"
    except Exception as e:
        return f"Error fetching WB indicator {indicator_code}: {e}"


# --- Expanded Definitions (Knowledge Base) ---
definitions = {
    "gdp": "Gross Domestic Product (GDP) is the total monetary or market value of all the finished goods and services produced within a country's borders in a specific time period.",
    "real gdp": "Real GDP is an inflation-adjusted measure that reflects the value of all goods and services produced by an economy in a given year, expressed in base-year prices.",
    "inflation": "Inflation is the rate at which the general level of prices for goods and services is rising, and subsequently, purchasing power is falling.",
    "cpi": "The Consumer Price Index (CPI) measures the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services.",
    "unemployment rate": "The unemployment rate is the percentage of the labor force that is jobless and actively looking for employment.",
    "federal funds rate": "The Federal Funds Rate is the target interest rate set by the Federal Open Market Committee (FOMC) at which commercial banks borrow and lend their excess reserves to each other overnight.",
    "fed funds rate": "The Federal Funds Rate is the target interest rate set by the Federal Open Market Committee (FOMC) at which commercial banks borrow and lend their excess reserves to each other overnight.",
    "10 year treasury rate": "The 10-Year Treasury Constant Maturity Rate is the yield on U.S. Treasury securities with a constant maturity of 10 years. It's often used as a benchmark for mortgage rates and other long-term borrowing costs.",
    "trade balance": "The Trade Balance (Goods) measures the difference between the monetary value of a nation's exports and imports of physical goods over a certain time period.",
    "industrial production": "The Industrial Production Index measures the real output of all relevant establishments located in the United States, regardless of their ownership, but not those located in U.S. territories.",
    "m2": "M2 is a measure of the U.S. money stock that includes M1 (currency and coins held by the non-bank public, checkable deposits, and travelers' checks) plus savings deposits, small-denomination time deposits, and balances in retail money market mutual funds."
}

# --- Expanded Indicator Mapping ---
indicator_map = {
    # Keyword(s) : {source, id, (optional) description for clarity}
    "real gdp": {"source": "fred", "id": "GDPC1", "desc": "Real Gross Domestic Product (Quarterly)"},
    "unemployment rate": {"source": "fred", "id": "UNRATE", "desc": "Unemployment Rate (Monthly)"},
    "cpi": {"source": "fred", "id": "CPIAUCSL_PC1", "desc": "Consumer Price Index % Change (Monthly)"},
    "inflation": {"source": "fred", "id": "CPIAUCSL_PC1", "desc": "Consumer Price Index % Change (Monthly)"}, # Map inflation to CPI % change
    "federal funds rate": {"source": "fred", "id": "FEDFUNDS", "desc": "Effective Federal Funds Rate (Monthly Avg)"},
    "fed funds rate": {"source": "fred", "id": "FEDFUNDS", "desc": "Effective Federal Funds Rate (Monthly Avg)"},
    "10 year treasury rate": {"source": "fred", "id": "GS10", "desc": "10-Year Treasury Constant Maturity Rate (Monthly Avg)"},
    "trade balance": {"source": "fred", "id": "BOPGTB", "desc": "Trade Balance - Goods (Monthly)"},
    "industrial production": {"source": "fred", "id": "INDPRO", "desc": "Industrial Production Index (Monthly)"},
    "m2": {"source": "fred", "id": "M2SL", "desc": "M2 Money Stock (Monthly)"},
    # Add World Bank examples if needed (adjust function calls)
    # "gdp world bank": {"source": "worldbank", "id": "NY.GDP.MKTP.CD", "country": "US", "desc": "Nominal GDP (Current US$)"}
}

def get_definition(indicator_keyword):
    """Retrieves the definition for a keyword."""
    # Find the best matching key (simple prefix check)
    key_found = None
    for key in definitions.keys():
         if indicator_keyword.lower() in key:
              key_found = key
              break
    return definitions.get(key_found, f"Sorry, I don't have a definition for '{indicator_keyword}'.")


# Example Usage (for testing)
if __name__ == "__main__":
    print("--- Testing Data Fetcher ---")
    if fred: # Only test if FRED API is initialized
        print("Testing FRED Real GDP:", get_fred_series_latest('GDPC1'))
        print("Testing FRED Unemployment:", get_fred_series_latest('UNRATE'))
        print("Testing FRED Fed Funds Rate:", get_fred_series_latest('FEDFUNDS'))
        print("Testing FRED Trade Balance:", get_fred_series_latest('BOPGTB'))
    else:
        print("FRED API not initialized, skipping FRED tests.")

    print("Testing GDP Definition:", get_definition('gdp'))
    print("Testing M2 Definition:", get_definition('m2'))
    print("--- End Testing ---")