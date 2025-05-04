# app.py (Combined CSV Analysis + Gemini LLM)
import streamlit as st
import pandas as pd
import numpy as np
import random
import re
import nltk
import os

# Import analysis libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr

# --- 1. Set Page Config FIRST ---
st.set_page_config(page_title="Analytical Chatbot", page_icon="🧠")

# --- 2. Gemini LLM Setup ---
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        try: GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        except: pass # Keep going if not found, will show warning later

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        genai_configured = True
        # st.toast("Gemini LLM Initialized.", icon="✅") # Optional confirmation
    else:
        st.warning("GEMINI_API_KEY not found. LLM features (explanation, fallback) will be disabled.", icon="⚠️")
        genai_configured = False
        gemini_model = None

except ImportError:
    st.warning("Google Generative AI library not found. LLM features disabled. Install with: pip install google-generativeai", icon="⚠️")
    genai_configured = False
    gemini_model = None
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}", icon="🚨")
    genai_configured = False
    gemini_model = None
# --- End Gemini Setup ---

# --- 3. Define Functions (Checks, Loading, Preparation, Analysis, Gemini) ---

# NLTK Check Function
def check_nltk_resources():
    messages = []
    try: nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError: messages.append("Downloading NLTK 'punkt'..."); nltk.download('punkt', quiet=True)
    try: nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError: messages.append("Downloading NLTK 'stopwords'..."); nltk.download('stopwords', quiet=True)
    return messages

# Load Data Function
@st.cache_data
def load_data():
    try:
        churn_df = pd.read_csv('Invol_churn_channel_wise_actuals_and_forecast.csv')
        macro_df = pd.read_csv('macroeconomic_data.csv')
        if churn_df.empty or macro_df.empty: return None, None, "Error: CSV files are empty."
        # Basic validation: check expected columns exist
        expected_churn_cols = ['Date', 'Sales Channel', 'Customer Tenure', 'Invol_Churn_Value']
        expected_macro_cols = ['Date', 'Macroeconomic Indicator Name', 'Value']
        if not all(col in churn_df.columns for col in expected_churn_cols): return None, None, f"Churn CSV missing expected columns ({expected_churn_cols})."
        if not all(col in macro_df.columns for col in expected_macro_cols): return None, None, f"Macro CSV missing expected columns ({expected_macro_cols})."
        return churn_df, macro_df, None
    except FileNotFoundError: return None, None, "Error: Ensure CSV files are present."
    except pd.errors.EmptyDataError: return None, None, "Error: One or both CSV files are empty/invalid."
    except Exception as e: return None, None, f"Error loading data: {e}"

# Data Preparation Function
@st.cache_data
def prepare_analysis_data(churn_df, macro_df):
    try:
        churn_agg = churn_df.groupby('Date')['Invol_Churn_Value'].sum().reset_index()
        churn_agg = churn_agg.rename(columns={'Invol_Churn_Value': 'Total_Churn'})
        macro_pivot = macro_df.pivot_table(index='Date', columns='Macroeconomic Indicator Name', values='Value').reset_index()
        analysis_df = pd.merge(churn_agg, macro_pivot, on='Date', how='inner')
        # Convert Date to string to ensure consistent type after potential pivoting/merging
        analysis_df['Date'] = analysis_df['Date'].astype(str)
        if analysis_df.empty: return None, "Error: No common dates after merging."
        # Handle potential NaNs created during pivot/merge if necessary (e.g., ffill or check model handling)
        # analysis_df = analysis_df.ffill().bfill() # Example: fill NaNs
        return analysis_df, None
    except Exception as e: return None, f"Error preparing data for analysis: {e}"

# --- Analysis Functions (calculate_correlation, plot_trend, analyze_impact, format_value) ---
# Keep these functions as defined in the previous (CSV Analysis) step.
# Make sure they accept the analysis_df and necessary parameters.
# Example stubs (replace with full functions from previous answer):
def format_value(value, indicator_name=None):
    # ... (full implementation) ...
    if pd.isna(value): return "N/A"
    if indicator_name and ('rate' in indicator_name.lower() or 'gdp adjusted' in indicator_name.lower()): return f"{value:.1f}%"
    if abs(value) >= 1_000_000: return f"{value:,.0f}"
    if abs(value) >= 1_000: return f"{value:,.0f}"
    return f"{value:.1f}"

def calculate_correlation(df, target_var='Total_Churn', indicators=None, date_range=None):
    # ... (full implementation) ...
    if df is None or target_var not in df.columns: return None, f"Target variable '{target_var}' not found."
    df_filtered = df.copy()
    if date_range and len(date_range) == 2:
         try: df_filtered = df_filtered[(df_filtered['Date'] >= date_range[0]) & (df_filtered['Date'] <= date_range[1])]
         except Exception as e: return None, f"Error applying date range: {e}"
    if df_filtered.empty or len(df_filtered) < 2: return None, "Not enough data in range for correlation."
    if not indicators: indicators = df_filtered.select_dtypes(include=np.number).columns.tolist(); indicators.remove(target_var)
    valid_indicators = [ind for ind in indicators if ind in df_filtered.columns]; cols_to_correlate = [target_var] + valid_indicators
    if not valid_indicators: return None, "Specified indicators not found."
    try:
        correlation_matrix = df_filtered[cols_to_correlate].corr(method='pearson')
        if correlation_matrix.empty or target_var not in correlation_matrix: return None, "Could not calculate correlations."
        target_correlations = correlation_matrix[target_var].drop(target_var)
        return target_correlations.sort_values(ascending=False), None
    except Exception as e: return None, f"Error calculating correlation: {e}"


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


def analyze_impact(df, target_var='Total_Churn', predictor_vars=None, date_range=None):
     # ... (full implementation) ...
    if df is None or target_var not in df.columns: return None, f"Target '{target_var}' not found."
    df_filtered = df.copy()
    if date_range and len(date_range) == 2: df_filtered = df_filtered[(df_filtered['Date'] >= date_range[0]) & (df_filtered['Date'] <= date_range[1])]
    if df_filtered.empty or len(df_filtered) < 5: return None, "Not enough data in range."
    if not predictor_vars: predictor_vars = df_filtered.select_dtypes(include=np.number).columns.tolist(); predictor_vars.remove(target_var)
    valid_predictors = [p for p in predictor_vars if p in df_filtered.columns]
    if not valid_predictors: return None, "Specified predictors not found."
    cols_for_model = [target_var] + valid_predictors; df_model_data = df_filtered[cols_for_model].dropna()
    if len(df_model_data) < len(valid_predictors) + 2: return None, "Not enough valid data points after handling missing values."
    Y = df_model_data[target_var]; X = df_model_data[valid_predictors]; X = sm.add_constant(X)
    try: model = sm.OLS(Y, X).fit(); return model.summary(), None
    except Exception as e: return None, f"Regression error: {e}"

# --- Gemini Interaction Function ---
def ask_gemini(prompt):
    """Sends a prompt to the configured Gemini model."""
    if not genai_configured or not gemini_model:
        return "My advanced knowledge module (Gemini LLM) is not configured or available."
    st.info("Asking Gemini...", icon="🧠")
    try:
        response = gemini_model.generate_content(prompt)
        if response.parts: return response.text
        else: st.warning(f"Gemini response issue: {response.prompt_feedback}", icon="⚠️"); return "I received an unusual response from my advanced module."
    except Exception as e: st.error(f"Error calling Gemini API: {e}", icon="🚨"); return "Sorry, I encountered an error connecting to my advanced knowledge module."

# --- Parsing Function (with inflation mapping) ---
def parse_query_v3(text, macro_indicators_list, all_analysis_cols):
    """Enhanced parser for analysis, lookup, explain intents + inflation mapping."""
    text_lower = text.lower()
    params = {'intent': 'unknown', 'clarification': None}

    # Date Range Parsing (same as before)
    # ... (keep date parsing logic) ...
    date_matches = re.findall(r'(?:(q[1-4])[ -]?)?(\d{4})', text_lower)
    dates = []
    for q, y in date_matches:
        if q: dates.append(f"{y}-{q.upper()}")
        else: dates.extend([f"{y}-Q1", f"{y}-Q2", f"{y}-Q3", f"{y}-Q4"])
    if 'between' in text_lower and len(dates) >= 2: params['date_range'] = sorted(list(set(dates)))[:2]
    elif 'from' in text_lower and 'to' in text_lower and len(dates) >= 2: params['date_range'] = sorted(list(set(dates)))[:2]
    elif len(dates) == 1: params['date_range'] = [dates[0], dates[0]]


    # --- Intent Detection (Add 'explain') ---
    if 'correlation' in text_lower or 'correlate' in text_lower: params['intent'] = 'correlation'
    elif 'trend' in text_lower or 'show me' in text_lower: params['intent'] = 'trend'
    elif 'impact' in text_lower or 'affect' in text_lower or 'effect' in text_lower or 'influence' in text_lower: params['intent'] = 'impact'
    elif 'explain' in text_lower or 'tell me more' in text_lower or 'what is' in text_lower or 'define' in text_lower: params['intent'] = 'explain' # Gemini target
    elif 'latest' in text_lower: params['intent'] = 'latest_value'
    else: params['intent'] = 'lookup' # Default if specific keywords missing


    # --- Entity/Variable Extraction (including inflation mapping) ---
    params['target_variable'] = 'Total_Churn'
    predictors = []
    columns_to_plot = []
    explain_subject = None # What to explain?
    found_specific_macro = False

    if 'churn' in text_lower:
         params['type'] = params.get('type', 'churn')
         if params['intent'] == 'trend': columns_to_plot.append('Total_Churn')
         if params['intent'] == 'explain': explain_subject = 'Total Churn'

    macro_indicators_lower = {m.lower(): m for m in macro_indicators_list} if macro_indicators_list else {}
    for indicator_lower in sorted(macro_indicators_lower.keys(), key=len, reverse=True):
        if indicator_lower in text_lower:
             params['type'] = 'macro'
             indicator_original = macro_indicators_lower[indicator_lower]
             predictors.append(indicator_original)
             if params['intent'] == 'trend': columns_to_plot.append(indicator_original)
             if params['intent'] == 'lookup': params['lookup_indicator'] = indicator_original
             if params['intent'] == 'explain': explain_subject = indicator_original # Prioritize specific indicator for explain
             found_specific_macro = True

    if not found_specific_macro and 'inflation' in text_lower:
        proxy_indicator = "gdp adjusted for inflation"
        if proxy_indicator in macro_indicators_lower.values():
            params['type'] = 'macro'
            predictors.append(proxy_indicator)
            if params['intent'] == 'trend': columns_to_plot.append(proxy_indicator)
            if params['intent'] == 'lookup': params['lookup_indicator'] = proxy_indicator
            if params['intent'] == 'explain': explain_subject = proxy_indicator # Explain the proxy
            params['clarification'] = f"(Note: Using '{proxy_indicator}' as proxy for 'inflation'.)"
            found_specific_macro = True

    # If intent is explain but no subject identified, maybe explain the whole topic?
    if params['intent'] == 'explain' and not explain_subject:
         if 'data' in text_lower or 'analysis' in text_lower: explain_subject = 'the available data analysis'
         else: explain_subject = text # Fallback to explaining the raw text? Or ask for clarification? Let's assume fallback needed.

    params['predictors'] = list(set(predictors))
    params['plot_columns'] = list(set(columns_to_plot))
    params['explain_subject'] = explain_subject

    # Handle 'latest' intent processing
    if params['intent'] == 'latest_value':
         params['intent'] = 'lookup'; params['latest'] = True
         if not params.get('lookup_indicator'):
              if 'churn' in text_lower: params['lookup_indicator'] = 'Total_Churn'
              elif params.get('type') == 'macro': params['intent'] = 'clarify_indicator'

    if params['intent'] == 'trend' and not params['plot_columns']: params['plot_columns'] = [params['target_variable']]

    return params

# --- Combined Bot Response Function ---
def get_bot_response_v3(user_input, analysis_df, macro_names):
    """Orchestrates CSV analysis and Gemini LLM calls."""
    if analysis_df is None: return "Sorry, the analysis data isn't available."

    all_analysis_cols = analysis_df.columns.tolist() if analysis_df is not None else []
    params = parse_query_v3(user_input, macro_names, all_analysis_cols)
    intent = params.get('intent')
    date_range = params.get('date_range')
    target_var = params.get('target_variable', 'Total_Churn')
    predictors = params.get('predictors')
    plot_columns = params.get('plot_columns')
    lookup_indicator = params.get('lookup_indicator')
    fetch_latest = params.get('latest', False)
    explain_subject = params.get('explain_subject')
    clarification = params.get('clarification')

    disclaimer = "\n\n*(Note: Analysis uses dummy data. Correlation/association doesn't imply causation.)*"
    response = None
    figure = None # To hold potential plot object

    # --- Handle Specific Intents ---
    if intent == 'correlation':
        st.info("Calculating correlations...", icon="🔗")
        target_correlations, error = calculate_correlation(analysis_df, target_var, predictors, date_range)
        if error: response = error
        elif target_correlations is not None and not target_correlations.empty:
            response_text = f"**Correlation with {target_var}**"
            if date_range: response_text += f" ({date_range[0]} to {date_range[1]})"
            response_text += ":\n" + "\n".join([f"- {idx}: {val:.2f}" for idx, val in target_correlations.items()])
            response = response_text + disclaimer
        else: response = "Could not calculate correlations."

    elif intent == 'trend':
         if not plot_columns: response = "Please specify what trend to show."
         else:
              st.info(f"Generating trend plot...", icon="📈")
              fig, error = plot_trend(analysis_df, plot_columns, date_range, title=f"Trend for {', '.join(plot_columns)}")
              if error: response = error
              elif fig:
                  figure = fig # Store figure to display later
                  response = f"Here's the trend plot for {', '.join(plot_columns)}." + disclaimer
              else: response = "Could not generate trend plot."

    elif intent == 'impact':
         if not predictors: response = "Please specify which indicator(s) to analyze the impact of."
         else:
              st.info(f"Analyzing impact of {', '.join(predictors)} on {target_var}...", icon="🔬")
              summary, error = analyze_impact(analysis_df, target_var, predictors, date_range)
              if error: response = error
              elif summary:
                  response_text = f"**Simple Impact Analysis Results (Regression)**\n"
                  if date_range: response_text += f"*Period: {date_range[0]} to {date_range[1]}*\n"
                  st.text(summary) # Display full summary table
                  response = response_text + f"See regression summary above for details." + disclaimer
              else: response = "Could not perform impact analysis."

    elif intent == 'lookup':
         lookup_target = lookup_indicator or target_var
         target_date = None
         if fetch_latest: target_date = analysis_df['Date'].max()
         elif date_range: target_date = date_range[0]

         if not target_date: response = f"Please specify date or 'latest' for {lookup_target}."
         elif lookup_target not in analysis_df.columns: response = f"'{lookup_target}' not found."
         else:
             try:
                 value_series = analysis_df.loc[analysis_df['Date'] == target_date, lookup_target]
                 if value_series.empty: response = f"No data found for {lookup_target} in {target_date}."
                 else:
                     value = value_series.iloc[0]
                     response = f"The value for **{lookup_target}** in **{target_date}** was: {format_value(value, lookup_target)}"
             except Exception as e: response = f"Error retrieving lookup value: {e}"

    elif intent == 'explain':
         if not explain_subject:
              # If explain intent but no clear subject, fallback to general Gemini call
              intent = 'unknown' # Force fallback
         else:
              # Prepare context for Gemini explanation
              context = f"Explain the concept of '{explain_subject}' in a macroeconomic context."
              # Try to add latest value as context if relevant
              if explain_subject in analysis_df.columns:
                  latest_val_explain = analysis_df.loc[analysis_df['Date'] == analysis_df['Date'].max(), explain_subject]
                  if not latest_val_explain.empty:
                      latest_val = latest_val_explain.iloc[0]
                      context += f" The latest value in the dataset ({analysis_df['Date'].max()}) is {format_value(latest_val, explain_subject)}."
              context += " Keep the explanation concise for a chatbot."
              response = ask_gemini(context)


    elif intent == 'clarify_indicator':
          response = "Please specify which macroeconomic indicator you want the latest value for."

    # --- Fallback to Gemini if no specific response generated ---
    if response is None and intent == 'unknown':
         st.info("Trying to answer with Gemini...", icon="💡")
         response = ask_gemini(f"As a macroeconomic chatbot assistant, answer the following user query concisely based on general knowledge or inferring from the request context related to churn and macroeconomics: '{user_input}'")
         # If Gemini fails, use a standard unknown response
         if "Sorry, I encountered an error" in response or "module is not configured" in response:
              response = random.choice(UNKNOWN_RESPONSES)

    # --- Append clarification if it exists ---
    if clarification and response and "Sorry" not in response:
        response += f"\n{clarification}"

    return response, figure # Return response text and potential figure object


# --- 4. Execute Initial Checks and Load/Prepare Data ---
nltk_messages = check_nltk_resources()
churn_data_raw, macro_data_raw, load_error_message = load_data()

# Display Initial Warnings/Errors (AFTER set_page_config)
for msg in nltk_messages: st.toast(msg, icon="ℹ️")
if load_error_message: st.error(load_error_message, icon="🚨"); st.stop()

# Prepare Data for Analysis
analysis_data, prep_error_message = prepare_analysis_data(churn_data_raw, macro_data_raw)
if prep_error_message: st.error(prep_error_message, icon="🚨"); st.stop()

macro_indicator_names = macro_data_raw['Macroeconomic Indicator Name'].unique().tolist() if macro_data_raw is not None else []

# --- 5. Setup UI ---
st.title("🧠 Analytical Chatbot")
st.caption("Ask for trends, correlations, impact, definitions, or latest values.")

# Initialize/Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I analyze the data or provide explanations today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display plots stored in messages if they exist (modification needed if storing plots)
        # For now, plots are displayed directly when generated
        st.markdown(message["content"])

# --- 6. Handle User Input ---
if prompt := st.chat_input("Ask your analysis question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response (which might include displaying plots/summaries)
    with st.spinner("Thinking..."):
        bot_reply_text, generated_figure = get_bot_response_v3(prompt, analysis_data, macro_indicator_names)

    # Display bot's response text
    with st.chat_message("assistant"):
        st.markdown(bot_reply_text)
        # Display the figure if one was generated by plot_trend
        # Note: Impact analysis summary (st.text) is displayed directly inside get_bot_response_v3
        if generated_figure:
             st.pyplot(generated_figure)

    # Add text response to history (figure is not easily serializable for session state)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply_text})