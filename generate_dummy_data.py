# generate_dummy_data.py
import pandas as pd
import numpy as np
import itertools # To easily create combinations

# --- Configuration ---
start_date = '2021-01-01'
end_date = '2028-12-31'
output_churn_file = 'Invol_churn_channel_wise_actuals_and_forecast.csv'
output_macro_file = 'macroeconomic_data.csv'

# --- Generate Quarterly Dates ---
# Use pandas quarterly frequency (ends of quarters)
dates = pd.date_range(start=start_date, end=end_date, freq='Q')
# Format as YYYY-QQ (e.g., 2021-Q1)
quarterly_dates = [f"{d.year}-Q{d.quarter}" for d in dates]

# --- Generate Churn Data ---
sales_channels = ['digital', 'stores', 'national retailer']
# Correcting the tenure range slightly for consistency
customer_tenures = ['0-12Months', '13-24Months', '25-36Months']

# Create all combinations of date, channel, tenure
churn_combinations = list(itertools.product(quarterly_dates, sales_channels, customer_tenures))

# Create DataFrame
churn_df = pd.DataFrame(churn_combinations, columns=['Date', 'Sales Channel', 'Customer Tenure'])

# Generate dummy churn values
min_churn = 100000
max_churn = 180000
churn_df['Invol_Churn_Value'] = np.random.randint(min_churn, max_churn + 1, size=len(churn_df))

# Save the churn data CSV
churn_df.to_csv(output_churn_file, index=False)
print(f"Successfully created '{output_churn_file}' with {len(churn_df)} rows.")


# --- Generate Macroeconomic Data ---
macro_indicators = [
    'consumer credit',
    'business confidence',
    'gdp adjusted for inflation', # Typically a % change or index
    'unemployment rate',        # Typically a small %
    'financial liabilities',    # Typically a large number
    'savings rate'              # Typically a %
]

# Create combinations of date and indicator
macro_combinations = list(itertools.product(quarterly_dates, macro_indicators))

# Create DataFrame
macro_df = pd.DataFrame(macro_combinations, columns=['Date', 'Macroeconomic Indicator Name'])

# Generate dummy values with somewhat realistic ranges/patterns
values = []
for _, row in macro_df.iterrows():
    indicator = row['Macroeconomic Indicator Name']
    if indicator == 'consumer credit':
        # Large numbers, generally increasing trend with noise
        base = 3000000 + (pd.Period(row['Date']).ordinal - pd.Period('2021-Q1').ordinal) * 50000
        value = base + np.random.randint(-20000, 20000)
    elif indicator == 'business confidence':
        # Index around 100
        value = np.random.uniform(85.0, 115.0)
    elif indicator == 'gdp adjusted for inflation':
        # Quarterly % change (small range around 0)
        value = np.random.uniform(-1.0, 2.5)
    elif indicator == 'unemployment rate':
        # Percentage, small range
        value = np.random.uniform(3.5, 6.5)
    elif indicator == 'financial liabilities':
        # Large numbers, generally increasing trend
        base = 15000000 + (pd.Period(row['Date']).ordinal - pd.Period('2021-Q1').ordinal) * 100000
        value = base + np.random.randint(-50000, 50000)
    elif indicator == 'savings rate':
        # Percentage, typical range
        value = np.random.uniform(4.0, 12.0)
    else:
        value = np.random.uniform(0, 100) # Default fallback
    values.append(round(value, 2)) # Round to 2 decimal places for neatness

macro_df['Value'] = values # Used 'Value' instead of 'Invol_churn_value'

# Save the macroeconomic data CSV
macro_df.to_csv(output_macro_file, index=False)
print(f"Successfully created '{output_macro_file}' with {len(macro_df)} rows.")