# app.py (Persistent Filters + Context + Gemini LLM + Scaling)
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
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler # Import scaler

# --- 1. Set Page Config FIRST ---
st.set_page_config(page_title="Analytical Chatbot+", page_icon="🔬", layout="wide") # Use wide layout

# --- 2. Gemini LLM Setup ---
# ... (Keep the Gemini setup code exactly as before) ...
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        try:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
            if not GEMINI_API_KEY: raise KeyError
        except:
            # Don't show warning here, show it later if LLM is actually needed
            genai_configured = False; gemini_model = None
        else: # Key found in secrets
             genai.configure(api_key=GEMINI_API_KEY)
             gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
             genai_configured = True
    else: # Key found in environment variables
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        genai_configured = True
except ImportError:
    st.warning("Google Generative AI library not found. LLM features disabled.", icon="⚠️")
    genai_configured = False; gemini_model = None
except Exception as e:
    st.error(f"Error initializing Gemini API: {e}", icon="🚨")
    genai_configured = False; gemini_model = None
# --- End Gemini Setup ---


# --- 3. Define Functions ---

# NLTK Check Function
def check_nltk_resources():
    # ... (Keep existing NLTK check function) ...
    messages = []
    try: nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError: messages.append("Downloading NLTK 'punkt'..."); nltk.download('punkt', quiet=True)
    return messages

# Load Data Function
@st.cache_data
def load_data():
    # ... (Keep existing load_data function) ...
    try:
        churn_df = pd.read_csv('Invol_churn_channel_wise_actuals_and_forecast.csv')
        macro_df = pd.read_csv('macroeconomic_data.csv')
        if churn_df.empty or macro_df.empty: return None, None, "Error: CSV files are empty."
        expected_churn_cols = ['Date', 'Sales Channel', 'Customer Tenure', 'Invol_Churn_Value']
        expected_macro_cols = ['Date', 'Macroeconomic Indicator Name', 'Value']
        if not all(col in churn_df.columns for col in expected_churn_cols): return None, None, f"Churn CSV missing columns."
        if not all(col in macro_df.columns for col in expected_macro_cols): return None, None, f"Macro CSV missing columns."
        return churn_df, macro_df, None
    except Exception as e: return None, None, f"Error loading data: {e}"


# Channel-Level Data Preparation Function
@st.cache_data
def prepare_channel_analysis_data(churn_df, macro_df):
    # ... (Keep existing prepare_channel_analysis_data function) ...
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


# --- Analysis Functions ---
def format_value(value, indicator_name=None):
    # ... (Keep existing implementation) ...
    if pd.isna(value): return "N/A"
    if indicator_name and ('rate' in indicator_name.lower() or 'gdp adjusted' in indicator_name.lower()): return f"{value:.1f}%"
    if abs(value) >= 1_000_000: return f"{value:,.0f}"
    if abs(value) >= 1_000: return f"{value:,.0f}"
    return f"{value:.1f}"

# --- plot_trend with Scaling ---
def plot_trend(df, column_names, date_range=None, title="Trend Analysis"):
    # ... (Keep existing implementation) ...
    if df is None: return None, "Data not available."
    df_filtered = df.copy()
    if date_range and len(date_range) == 2: df_filtered = df_filtered[(df_filtered['Date'] >= date_range[0]) & (df_filtered['Date'] <= date_range[1])]
    if df_filtered.empty: return None, "No data in the specified range to plot."
    valid_columns = [col for col in column_names if col in df_filtered.columns]
    if not valid_columns: return None, f"None of the columns {column_names} found in data."
    plot_data = df_filtered[['Date'] + valid_columns].copy()
    scaler = MinMaxScaler()
    numeric_cols = plot_data.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0: plot_data[numeric_cols] = scaler.fit_transform(plot_data[numeric_cols]); scaled = True
    else: scaled = False
    try: plot_data_melted = pd.melt(plot_data, id_vars=['Date'], var_name='Indicator', value_name='Scaled Value' if scaled else 'Value')
    except Exception as e: return None, f"Error reshaping data for plotting: {e}"
    fig, ax = plt.subplots(figsize=(10, 5)) # Adjust figsize if needed
    sns.lineplot(x='Date', y='Scaled Value' if scaled else 'Value', hue='Indicator', data=plot_data_melted, ax=ax, marker='o')
    plot_title = title; y_label = "Value"
    if scaled: plot_title += " (Scaled)"; y_label = "Scaled Value (Min-Max)"
    ax.set_title(plot_title); ax.set_xlabel("Quarter"); ax.set_ylabel(y_label); plt.xticks(rotation=45)
    ax.legend(title='Indicator', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
    return fig, None


def analyze_impact(df, target_var, predictor_vars, date_range=None):
     # ... (Keep existing implementation using statsmodels) ...
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


# --- Gemini Interaction Function ---
def ask_gemini(prompt, context=None):
    """Sends a prompt to Gemini, optionally including context with clearer instructions."""
    # ... (Keep existing implementation) ...
    if not genai_configured or not gemini_model:
        # Check if warning needs to be displayed now
        if 'gemini_warning_shown' not in st.session_state:
            st.warning("GEMINI_API_KEY not found or invalid. LLM features disabled.", icon="⚠️")
            st.session_state.gemini_warning_shown = True # Show only once per session
        return "My advanced knowledge module (LLM) is not available."

    if context:
        full_prompt = f"""Here is the summary of a previous data analysis:
<analysis_summary>
{context}
</analysis_summary>

Based *only* on the information in the analysis summary above, answer the following user query concisely: "{prompt}"

If the query cannot be answered from the summary, state that the information is not available in the summary. Do not use external knowledge unless the query explicitly asks for it or is unrelated to the summary."""
    else:
        full_prompt = f"As an analytical chatbot assistant, answer the following user query concisely: '{prompt}'"
    try:
        response = gemini_model.generate_content(full_prompt)
        if response.parts: return response.text
        else: st.warning(f"Gemini response issue: {response.prompt_feedback}", icon="⚠️"); return "I received an unusual response from my advanced module."
    except Exception as e: st.error(f"Error calling Gemini API: {e}", icon="🚨"); return "Sorry, error connecting to advanced knowledge module."


# --- Parsing Function (Simplified - Intent detection for chat) ---
def parse_chat_intent(text):
    """Detects primary intent for chat interaction (explain, summarize, unknown)."""
    text_lower = text.lower()
    # Removed 'impact' as it's now handled by the form
    if 'explain' in text_lower or 'tell me more' in text_lower or 'what is' in text_lower or 'define' in text_lower:
        return 'explain'
    if 'summarize' in text_lower or 'summary' in text_lower:
        return 'summarize'
    # Could add other intents here if needed
    return 'unknown' # Default for Gemini fallback

# --- Detailed Impact Analysis Function (Called by Form) ---
def perform_detailed_impact_analysis(df, target_churn_col, indicators, start_q, end_q):
    """Performs correlation and regression, returns results including text summary."""
    # ... (Keep existing implementation that generates 'text_summary') ...
    results = {'text_summary': f"Analysis Summary for {target_churn_col} ({start_q} to {end_q}):\n"}
    errors = []
    date_range = [start_q, end_q]
    correlations = None
    if target_churn_col not in df.columns: errors.append(f"Target churn column '{target_churn_col}' not found.")
    else:
        numeric_cols = [target_churn_col] + indicators
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
        if len(df_numeric) > 1:
            correlations = df_numeric.corr(method='pearson')[target_churn_col].drop(target_churn_col).sort_values(ascending=False)
            results['correlations'] = correlations
            if not correlations.empty:
                top_pos = correlations.head(1); top_neg = correlations.tail(1)
                results['text_summary'] += f"- Strongest positive correlation: {top_pos.index[0]} ({top_pos.iloc[0]:.2f})\n"
                results['text_summary'] += f"- Strongest negative correlation: {top_neg.index[0]} ({top_neg.iloc[0]:.2f})\n"
        else: errors.append("Not enough valid numeric data for correlation.")
    summary_obj, impact_error = analyze_impact(df, target_churn_col, indicators, date_range)
    if impact_error: errors.append(f"Impact Analysis Error: {impact_error}")
    results['regression_summary_obj'] = summary_obj
    if summary_obj:
        try:
            r_squared = float(summary_obj.tables[0].data[0][3]); adj_r_squared = float(summary_obj.tables[0].data[1][3])
            results['text_summary'] += f"- Regression Fit: R-squared={r_squared:.3f}, Adj. R-squared={adj_r_squared:.3f}\n"
            coeffs_df = pd.read_html(summary_obj.tables[1].as_html(), header=0, index_col=0)[0]
            significant_predictors = coeffs_df[coeffs_df['P>|t|'] < 0.05].index.tolist()
            if 'const' in significant_predictors: significant_predictors.remove('const')
            if significant_predictors: results['text_summary'] += f"- Significant predictors (p<0.05): {', '.join(significant_predictors)}\n"
            else: results['text_summary'] += "- No predictors found statistically significant (p<0.05).\n"
        except Exception as e: results['text_summary'] += f"- Could not parse regression summary details: {e}\n"
    else: results['text_summary'] += "- Regression analysis could not be completed.\n"
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

# Extract available columns for widgets (handle case where data prep failed)
if analysis_data is not None:
    churn_channels = [col.replace('Churn_', '') for col in analysis_data.columns if col.startswith('Churn_')]
    macro_indicators_list = [col for col in analysis_data.columns if not col.startswith('Churn_') and col != 'Date']
    available_quarters = sorted(analysis_data['Date'].unique())
else: # Set defaults if data prep failed to avoid errors rendering widgets
    churn_channels = []
    macro_indicators_list = []
    available_quarters = []


# --- 5. Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Use the filters above to run an impact analysis, then ask follow-up questions."}]
# Removed 'analysis_requested' state
if "analysis_results" not in st.session_state: # Holds the full dict including summary text
    st.session_state.analysis_results = None

# --- 6. Setup UI ---
st.title("🔬 Analytical Chatbot + LLM")
st.caption("Configure and run impact analysis using the filters below. Then ask follow-up questions about the results.")

# --- Persistent Impact Analysis Configuration Form ---
st.markdown("---")
st.subheader("Impact Analysis Configuration")
with st.form("impact_analysis_form"):
    # Use columns for better layout
    col_target, col_preds = st.columns([1, 3])
    with col_target:
        selected_target_channel = st.selectbox("Target Churn Channel:",
                                               options=churn_channels,
                                               key="target_channel_select",
                                               index=0 if churn_channels else None, # Handle empty list
                                               disabled=not churn_channels) # Disable if no channels
    with col_preds:
        selected_indicators = st.multiselect("Predictor Macro Indicator(s):",
                                             options=macro_indicators_list,
                                             key="indicators_multi_select",
                                             disabled=not macro_indicators_list) # Disable if no indicators

    col_start, col_end = st.columns(2)
    with col_start:
        selected_start_q = st.selectbox("Start Quarter:",
                                        options=available_quarters,
                                        index=0 if available_quarters else None,
                                        key="start_q_select",
                                        disabled=not available_quarters)
    with col_end:
        selected_end_q = st.selectbox("End Quarter:",
                                      options=available_quarters,
                                      index=len(available_quarters)-1 if available_quarters else None,
                                      key="end_q_select",
                                      disabled=not available_quarters)

    submitted = st.form_submit_button("📊 Run Impact Analysis", disabled=not analysis_data is not None)

    if submitted:
        # Validation within the form submit logic
        if not selected_target_channel:
             st.warning("Please select a target churn channel.")
        elif not selected_indicators:
            st.warning("Please select at least one macro indicator.")
        elif not selected_start_q or not selected_end_q:
             st.warning("Please select a valid date range.")
        elif selected_start_q > selected_end_q:
            st.warning("Start Quarter cannot be after End Quarter.")
        else:
            with st.spinner("Performing detailed analysis..."):
                target_churn_col = f"Churn_{selected_target_channel}"
                # Store the entire results dictionary in session state
                st.session_state.analysis_results = perform_detailed_impact_analysis(
                    analysis_data, target_churn_col, selected_indicators, selected_start_q, selected_end_q
                )
            # Add a message to chat indicating analysis is complete
            st.session_state.messages.append({"role": "assistant", "content": f"Impact analysis complete for {target_churn_col} ({selected_start_q} - {selected_end_q}). Results are displayed below. Feel free to ask follow-up questions."})
            st.rerun() # Rerun to display results and updated chat

# --- Display Analysis Results ---
if st.session_state.analysis_results:
    st.markdown("---")
    st.subheader("Latest Impact Analysis Results")
    results = st.session_state.analysis_results

    # Display errors first
    if results.get('errors'):
        for error in results['errors']: st.error(error, icon="🚨")

    # Display Text Summary
    if results.get('text_summary'):
        st.markdown("**Analysis Summary:**")
        st.markdown(results['text_summary']) # Display the generated summary

    # Use columns for better layout of results
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        # Display Correlations
        if 'correlations' in results and results['correlations'] is not None and not results['correlations'].empty:
            st.markdown("**Correlations with Target:**")
            st.dataframe(results['correlations'].apply(lambda x: f"{x:.2f}"))

        # Display Plot
        if 'figure' in results and results['figure'] is not None:
            st.markdown("**Trend Plot (Scaled):**")
            st.pyplot(results['figure'])

    with res_col2:
        # Display Regression Summary Object
        if 'regression_summary_obj' in results and results['regression_summary_obj'] is not None:
            st.markdown("**Detailed Regression Summary:**")
            st.text(results['regression_summary_obj']) # Display full statsmodels summary

    st.markdown("*(Note: Analysis uses dummy data. Correlation/association doesn't imply causation.)*")
    # Results remain in state until overwritten by a new analysis run

# --- Chat Interface ---
st.markdown("---")
st.subheader("Ask Questions")

# Display chat messages
container = st.container() # Use a container for chat history
with container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle User Input via chat box
if prompt := st.chat_input("Ask follow-up questions about the analysis or general questions..."):
    # Add user message to chat history immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Parse intent for chat interaction
    intent = parse_chat_intent(prompt)
    bot_reply_content = None

    # Retrieve context from the latest analysis run (if any)
    analysis_context = st.session_state.analysis_results.get('text_summary') if st.session_state.analysis_results else None

    # --- Generate Bot Response ---
    with st.spinner("Thinking..."):
        if intent == 'explain':
            gemini_prompt = f"Explain the following concept in the context of business/economics: '{prompt}'. Keep it concise for a chatbot."
            bot_reply_content = ask_gemini(gemini_prompt, context=analysis_context) # Pass context

        elif intent == 'summarize':
             if analysis_context:
                   gemini_prompt = "Summarize the key findings from the previous analysis."
                   bot_reply_content = ask_gemini(gemini_prompt, context=analysis_context) # Pass context
             else:
                  bot_reply_content = "No analysis results are currently available to summarize. Please run an impact analysis using the filters above first."

        else: # Fallback for 'unknown' intent
            gemini_prompt = f"As an analytical chatbot assistant, answer the following user query: '{prompt}'"
            bot_reply_content = ask_gemini(gemini_prompt, context=analysis_context) # Pass context

    # Add assistant response to chat history
    if bot_reply_content:
        st.session_state.messages.append({"role": "assistant", "content": bot_reply_content})

    # Rerun to display the updated chat immediately
    st.rerun()

