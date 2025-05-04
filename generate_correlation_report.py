# generate_correlation_report.py
import pandas as pd
import numpy as np
import io
import openpyxl # Needed for Excel writing

# --- Configuration ---
INPUT_CHURN_FILE = 'Invol_churn_channel_wise_actuals_and_forecast.csv'
INPUT_MACRO_FILE = 'macroeconomic_data.csv'
OUTPUT_CORRELATION_FILE = 'channel_macro_correlations.xlsx'

# --- Helper Functions ---

def load_source_data(churn_file, macro_file):
    """Loads the source CSV files."""
    try:
        churn_df = pd.read_csv(churn_file)
        macro_df = pd.read_csv(macro_file)
        if churn_df.empty or macro_df.empty:
            print("Error: One or both source CSV files are empty.")
            return None, None
        # Basic validation
        expected_churn_cols = ['Date', 'Sales Channel', 'Customer Tenure', 'Invol_Churn_Value']
        expected_macro_cols = ['Date', 'Macroeconomic Indicator Name', 'Value']
        if not all(col in churn_df.columns for col in expected_churn_cols):
            print(f"Error: Churn CSV missing expected columns.")
            return None, None
        if not all(col in macro_df.columns for col in expected_macro_cols):
            print(f"Error: Macro CSV missing expected columns.")
            return None, None
        print("Source CSV files loaded successfully.")
        return churn_df, macro_df
    except FileNotFoundError:
        print(f"Error: Make sure '{churn_file}' and '{macro_file}' exist.")
        return None, None
    except Exception as e:
        print(f"Error loading source data: {e}")
        return None, None

def prepare_channel_analysis_data(churn_df, macro_df):
    """Aggregates churn per channel, pivots macro data, and merges."""
    if churn_df is None or macro_df is None:
        return None
    try:
        # Aggregate Churn per Channel per Date
        churn_agg = churn_df.groupby(['Date', 'Sales Channel'])['Invol_Churn_Value'].sum().reset_index()

        # Pivot Churn Data
        churn_pivot = churn_agg.pivot_table(index='Date',
                                            columns='Sales Channel',
                                            values='Invol_Churn_Value').reset_index()
        churn_pivot.columns = ['Date'] + ['Churn_' + col for col in churn_pivot.columns[1:]]

        # Pivot Macro Data
        macro_pivot = macro_df.pivot_table(index='Date',
                                            columns='Macroeconomic Indicator Name',
                                            values='Value').reset_index()

        # Merge Data
        analysis_df = pd.merge(churn_pivot, macro_pivot, on='Date', how='inner')
        analysis_df['Date'] = analysis_df['Date'].astype(str)

        # Handle potential NaNs - crucial for correlation
        analysis_df = analysis_df.ffill().bfill() # Example fill strategy
        if analysis_df.isnull().any().any():
             print("Warning: Missing values remain after attempting to fill. Correlations might be affected.")

        if analysis_df.empty:
            print("Error: No common dates found after merging.")
            return None

        print("Channel-level analysis data prepared.")
        return analysis_df
    except Exception as e:
        print(f"Error preparing channel analysis data: {e}")
        return None

def calculate_and_save_correlations(df, filename):
    """Calculates Pearson & Spearman correlations and saves to Excel."""
    if df is None:
        print("Error: Analysis data not available.")
        return

    try:
        # Ensure only numeric columns are used for correlation
        numeric_df = df.select_dtypes(include=np.number)

        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pearson_corr.to_excel(writer, sheet_name='Pearson Correlation')
            spearman_corr.to_excel(writer, sheet_name='Spearman Correlation')

        print(f"Correlation report saved successfully to '{filename}'.")
    except Exception as e:
        print(f"Error calculating or saving correlations: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting correlation report generation...")
    churn_data, macro_data = load_source_data(INPUT_CHURN_FILE, INPUT_MACRO_FILE)

    if churn_data is not None and macro_data is not None:
        analysis_dataframe = prepare_channel_analysis_data(churn_data, macro_data)
        if analysis_dataframe is not None:
            calculate_and_save_correlations(analysis_dataframe, OUTPUT_CORRELATION_FILE)
    print("Correlation report generation finished.")

