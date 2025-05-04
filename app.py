# app.py (Impact Analysis + Gemini LLM)
import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import nltk
import os
import io

# Import analysis libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr # Keep for potential use in impact analysis display

# --- 1. Set Page Config FIRST ---
st.set_page_config(page_title="Analytical Chatbot+", page_icon="🔬")

# --- 2. Gemini LLM Setup ---
# This section is explicitly kept as requested
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        try:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            if not GEMINI_API_KEY: raise KeyError # Trigger exception if secret exists but is empty
        except:
            st.warning("GEMINI_API_KEY not found. LLM features (explanation, fallback) disabled.", icon="⚠️")
            genai_configured = False
            gemini_model = None
        else: # Key found in secrets
             genai.configure(api_key=GEMINI_API_KEY)
             gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
             genai_configured = True
    else: # Key found in environment variables
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        genai_configured = True

    if genai_configured:
        st.toast("Gemini LLM Initialized.", icon="✅") # Optional confirmation on load

except ImportError:
    st.warning("Google Generative AI library not found. LLM features disabled.", icon="⚠️")
    genai_configured = False
    gemini_model = None
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}", icon="🚨")
    genai_configured = False
    gemini_model = None
# --- End Gemini Setup ---


# --- 3. Define Functions ---

# NLTK Check Function
def check_nltk_resources():
    messages = []
    try: nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError: messages.append("Downloading NLTK 'punkt'..."); nltk.download('punkt', quiet=True)
    # Add other checks if needed
    return messages

# Load Data Function
@st.cache_data
def load_data():
    try:
        churn_df = pd.read_csv('Invol_churn_channel_wise_actuals_and_forecast.csv')
        macro_df = pd.read_csv('macroeconomic_data.csv')
        if churn_df.empty or macro_df.empty: return None, None, "Error: CSV files are empty."
        # Basic validation
        expected_churn_cols = ['Date', 'Sales Channel', 'Customer Tenure', 'Invol_Churn_Value']
        expected_macro_cols = ['Date', 'Macroeconomic Indicator Name', 'Value']
        if not all(col in churn_df.columns for col in expected_churn_cols): return None, None, f"Churn CSV missing columns."
        if not all(col in macro_df.columns for col in expected_macro_cols): return None, None, f"Macro CSV missing columns."
        return churn_df, macro_df, None
    except Exception as e: return None, None, f"Error loading data: {e}"

# Channel-Level Data Preparation Function (Still needed for impact analysis)
@st.cache_data
def prepare_channel_analysis_data(churn_df, macro_df):
    """Aggregates churn per channel, pivots macro data, and merges."""
    if churn_df is None or macro_df is None: return None, "Source data not loaded."
    try:
        churn_agg = churn_df.groupby(['Date', 'Sales Channel'])['Invol_Churn_Value'].sum().reset_index()
        churn_pivot = churn_agg.pivot_table(index='Date', columns='Sales Channel', values='Invol_Churn_Value').reset_index()
        churn_pivot.columns = ['Date'] + ['Churn_' + col for col in churn_pivot.columns[1:]]
        macro_pivot = macro_df.pivot_table(index='Date', columns='Macroeconomic Indicator Name', values='Value').reset_index()
        analysis_df = pd.merge(churn_pivot, macro_pivot, on='Date', how='inner')
        analysis_df['Date'] = analysis_df['Date'].astype(str)
        analysis_df = analysis_df.ffill().bfill() # Handle NaNs
        if analysis_df.isnull().any().any(): return None, "Warning: Missing values remain after fill."
        if analysis_df.empty: return None, "Error: No common dates after merging."
        return analysis_df, None
    except Exception as e: return None, f"Error preparing channel analysis data: {e}"

# --- Analysis Functions (format_value, plot_trend, analyze_impact) ---
# Keep these functions exactly as defined in the previous answer (app_py_v_advanced_analysis)
# They are needed for the interactive impact analysis section.
def format_value(value, indicator_name=None):
    # ... (full implementation) ...
    if pd.isna(value): return "N/A"
    if indicator_name and ('rate' in indicator_name.lower() or 'gdp adjusted' in indicator_name.lower()): return f"{value:.1f}%"
    if abs(value) >= 1_000_000: return f"{value:,.0f}"
    if abs(value) >= 1_000: return f"{value:,.0f}"
    return f"{value:.1f}"

def plot_trend(df, column_names, date_range=None, title="Trend Analysis"):
    # ... (full implementation) ...
    if df is None: return None, "Data not available."
    df_filtered = df.copy()
    if date_range and len(date_range) == 2: df_filtered = df_filtered[(df_filtered['Date'] >= date_range[0]) & (df_filtered['Date'] <= date_range[1])]
    if df_filtered.empty: return None, "No data in range."
    valid_columns = [col for col in column_names if col in df_filtered.columns]
    if not valid_columns: return None, f"Columns {column_names} not found."
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in valid_columns: sns.lineplot(x='Date', y=col, data=df_filtered, ax=ax, label=col, marker='o')
    ax.set_title(title); ax.set_xlabel("Quarter"); ax.set_ylabel("Value"); plt.xticks(rotation=45); ax.legend(); plt.tight_layout()
    return fig, None

def analyze_impact(df, target_var, predictor_vars, date_range=None):
     # ... (full implementation using statsmodels) ...
    if df is None or target_var not in df.columns: return None, f"Target '{target_var}' not found."
    df_filtered = df.copy()
    if date_range and len(date_range) == 2: df_filtered = df_filtered[(df_filtered['Date'] >= date_range[0]) & (df_filtered['Date'] <= date_range[1])]
    if df_filtered.empty or len(df_filtered) < 5: return None, "Not enough data in range."
    if not predictor_vars: return None, "No predictors specified."
    valid_predictors = [p for p in predictor_vars if p in df_filtered.columns]
    if not valid_predictors: return None, "Specified predictors not found."
    cols_for_model = [target_var] + valid_predictors
    for col in cols_for_model: df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    df_model_data = df_filtered[cols_for_model].dropna()
    if len(df_model_data) < len(valid_predictors) + 2: return None, "Not enough valid data points."
    Y = df_model_data[target_var]; X = df_model_data[valid_predictors]; X = sm.add_constant(X)
    try: model = sm.OLS(Y, X).fit(); return model.summary(), None
    except Exception as e: return None, f"Regression error: {e}"

# --- Gemini Interaction Function (Kept) ---
def ask_gemini(prompt):
    """Sends a prompt to the configured Gemini model."""
    if not genai_configured or not gemini_model:
        return "My advanced knowledge module (Gemini LLM) is not configured or available."
    # st.info("Asking Gemini...", icon="🧠") # Optional: Can be noisy
    try:
        response = gemini_model.generate_content(prompt)
        if response.parts: return response.text
        else: st.warning(f"Gemini response issue: {response.prompt_feedback}", icon="⚠️"); return "I received an unusual response from my advanced module."
    except Exception as e: st.error(f"Error calling Gemini API: {e}", icon="🚨"); return "Sorry, error connecting to advanced knowledge module."

# --- Parsing Function (Identify 'impact' or 'explain' or fallback) ---
def parse_intent_v2(text):
    """Detects primary intent: impact, explain, or unknown."""
    text_lower = text.lower()
    if 'impact' in text_lower or 'affect' in text_lower or 'influence' in text_lower:
        return 'impact'
    if 'explain' in text_lower or 'tell me more' in text_lower or 'what is' in text_lower or 'define' in text_lower:
        return 'explain'
    # Could add 'trend', 'correlation' if direct parsing is desired later
    return 'unknown' # Default for Gemini fallback or simple lookup (if added)

# --- Detailed Impact Analysis Function (Called by Form) ---
def perform_detailed_impact_analysis(df, target_churn_col, indicators, start_q, end_q):
    """Performs correlation and regression for selected inputs."""
    results = {}
    errors = []
    date_range = [start_q, end_q]

    # 1. Calculate Correlations (using Pearson for simplicity here, can add Spearman)
    if target_churn_col not in df.columns:
        errors.append(f"Target churn column '{target_churn_col}' not found.")
    else:
        # Use only numeric columns for correlation
        numeric_cols = [target_churn_col] + indicators
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()

        if len(df_numeric) > 1:
            correlations = df_numeric.corr(method='pearson')[target_churn_col].drop(target_churn_col)
            results['correlations'] = correlations.sort_values(ascending=False)
        else:
             errors.append("Not enough valid numeric data for correlation.")


    # 2. Perform Regression Analysis
    summary, impact_error = analyze_impact(df, target_churn_col, indicators, date_range)
    if impact_error: errors.append(f"Impact Analysis Error: {impact_error}")
    results['regression_summary'] = summary

    # 3. Generate Trend Plot
    plot_cols = [target_churn_col] + indicators
    fig, plot_error = plot_trend(df, plot_cols, date_range, title=f"Trend for {target_churn_col} and Predictors")
    if plot_error: errors.append(f"Plotting Error: {plot_error}")
    results['figure'] = fig

    results['errors'] = errors
    return results

# --- 4. Execute Initial Checks and Load/Prepare Data ---
# nltk_messages = check_nltk_resources()
churn_data_raw, macro_data_raw, load_error_message = load_data()

# Display Initial Warnings/Errors (AFTER set_page_config)
# for msg in nltk_messages: st.toast(msg, icon="ℹ️")
if load_error_message: st.error(load_error_message, icon="🚨"); st.stop()

# Prepare Data for Analysis
analysis_data, prep_error_message = prepare_channel_analysis_data(churn_data_raw, macro_data_raw)
if prep_error_message: st.error(prep_error_message, icon="🚨"); st.stop()

# Extract available columns for widgets
churn_channels = [col.replace('Churn_', '') for col in analysis_data.columns if col.startswith('Churn_')]
macro_indicators_list = [col for col in analysis_data.columns if not col.startswith('Churn_') and col != 'Date']
available_quarters = sorted(analysis_data['Date'].unique())


# --- 5. Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me to explain concepts, or analyze the impact of macro indicators on channel churn."}]
if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 6. Setup UI ---
st.title("🔬 Analytical Chatbot + LLM")
st.caption("Analyze churn impacts using CSV data or ask general/explanation questions.")

# --- Main Chat Interface ---
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Impact Analysis Widget Section ---
if st.session_state.analysis_requested:
    st.markdown("---")
    st.subheader("Impact Analysis Configuration")
    with st.form("impact_analysis_form"):
        selected_target_channel = st.selectbox("Target Churn Channel:", options=churn_channels, key="target_channel_select")
        selected_indicators = st.multiselect("Predictor Macro Indicator(s):", options=macro_indicators_list, key="indicators_multi_select")
        col1, col2 = st.columns(2)
        with col1: selected_start_q = st.selectbox("Start Quarter:", options=available_quarters, index=0, key="start_q_select")
        with col2: selected_end_q = st.selectbox("End Quarter:", options=available_quarters, index=len(available_quarters)-1, key="end_q_select")
        submitted = st.form_submit_button("Run Impact Analysis")

        if submitted:
            if not selected_indicators: st.warning("Please select at least one macro indicator.")
            elif selected_start_q > selected_end_q: st.warning("Start Quarter cannot be after End Quarter.")
            else:
                with st.spinner("Performing detailed analysis..."):
                    target_churn_col = f"Churn_{selected_target_channel}"
                    analysis_results = perform_detailed_impact_analysis(analysis_data, target_churn_col, selected_indicators, selected_start_q, selected_end_q)
                    st.session_state.analysis_results = analysis_results
                    st.session_state.analysis_requested = False # Hide widgets after running
                    st.rerun()

# --- Display Analysis Results ---
if st.session_state.analysis_results:
    st.markdown("---")
    st.subheader("Impact Analysis Results")
    results = st.session_state.analysis_results
    if results.get('errors'):
        for error in results['errors']: st.error(error, icon="🚨")
    if 'correlations' in results and results['correlations'] is not None:
        st.markdown("**Correlations with Target:**")
        st.dataframe(results['correlations'].apply(lambda x: f"{x:.2f}"))
    if 'regression_summary' in results and results['regression_summary'] is not None:
        st.markdown("**Regression Summary:**")
        st.text(results['regression_summary'])
    if 'figure' in results and results['figure'] is not None:
        st.markdown("**Trend Plot:**")
        st.pyplot(results['figure'])
    st.markdown("*(Note: Analysis uses dummy data. Correlation/association doesn't imply causation.)*")
    # Clear results after displaying to prevent showing again on next interaction
    st.session_state.analysis_results = None


# --- Handle User Input ---
if prompt := st.chat_input("Ask about impact, explain concepts, or general questions..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Basic intent parsing
    intent = parse_intent_v2(prompt)
    bot_reply_content = None
    st.session_state.analysis_results = None # Clear previous analysis results

    if intent == 'impact':
        # Set flag to show widgets, provide instructions
        st.session_state.analysis_requested = True
        bot_reply_content = "Okay, please use the widgets below to configure the impact analysis."
        st.rerun() # Rerun to show widgets immediately

    elif intent == 'explain':
        # Use Gemini for explanations
        with st.spinner("Thinking..."):
            # Simple prompt for explanation
            gemini_prompt = f"Explain the following concept in the context of business/economics: '{prompt}'. Keep it concise for a chatbot."
            bot_reply_content = ask_gemini(gemini_prompt)

    else: # Fallback to Gemini for unknown intents or general questions
        with st.spinner("Thinking..."):
            # General purpose prompt
            gemini_prompt = f"As an analytical chatbot assistant, answer the following user query: '{prompt}'"
            bot_reply_content = ask_gemini(gemini_prompt)

    # Add assistant response to history and display (only if not rerunning for widgets)
    if bot_reply_content:
        st.session_state.messages.append({"role": "assistant", "content": bot_reply_content})
        with st.chat_message("assistant"):
             st.markdown(bot_reply_content)

