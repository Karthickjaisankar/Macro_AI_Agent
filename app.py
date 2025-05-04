# app.py (Persistent Filters + Detailed Context + Gemini LLM + Scaling - V4)
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
def ask_gemini(prompt, context=None, intent='unknown'):
    """
    Sends a prompt to Gemini.
    Prepends context as background if available for 'explain' or 'unknown' intents.
    Uses context exclusively for 'summarize' intent.
    """
    # ... (Keep existing implementation from V3) ...
    if not genai_configured or not gemini_model:
        if 'gemini_warning_shown' not in st.session_state:
            st.warning("GEMINI_API_KEY not found or invalid. LLM features disabled.", icon="⚠️")
            st.session_state.gemini_warning_shown = True
        return "My advanced knowledge module (LLM) is not available."

    if context and intent == 'summarize':
         full_prompt = f"""Here is the summary of a previous data analysis:
<analysis_summary>
{context}
</analysis_summary>
Based *only* on the analysis summary provided above, summarize the key findings concisely for the user."""
    elif context and (intent == 'explain' or intent == 'unknown'):
         full_prompt = f"""You can use the following summary of a previous data analysis as background context if relevant, but prioritize answering the user's main query using your general knowledge if the query is not directly about the analysis details:
<analysis_summary>
{context}
</analysis_summary>
User Query: "{prompt}"
Answer concisely as an analytical chatbot assistant."""
    else: # No context or intent doesn't use context in a special way
        full_prompt = f"As an analytical chatbot assistant, answer the following user query concisely: '{prompt}'"
    try:
        response = gemini_model.generate_content(full_prompt)
        if response.parts: return response.text
        else: st.warning(f"Gemini response issue: {response.prompt_feedback}", icon="⚠️"); return "I received an unusual response from my advanced module."
    except Exception as e: st.error(f"Error calling Gemini API: {e}", icon="🚨"); return "Sorry, error connecting to advanced knowledge module."


# --- Parsing Function (Simplified - Intent detection for chat) ---
def parse_chat_intent(text):
    # ... (Keep existing implementation) ...
    text_lower = text.lower()
    if 'explain' in text_lower or 'tell me more' in text_lower or 'what is' in text_lower or 'define' in text_lower: return 'explain'
    if 'summarize' in text_lower or 'summary' in text_lower: return 'summarize'
    return 'unknown'

# --- MODIFIED: Detailed Impact Analysis Function (More Detailed Summary) ---
def perform_detailed_impact_analysis(df, target_churn_col, indicators, start_q, end_q):
    """Performs correlation and regression, returns results including a more detailed text summary."""
    results = {'detailed_text_summary': f"**Detailed Analysis Report: {target_churn_col} ({start_q} to {end_q})**\n\n"}
    errors = []
    date_range = [start_q, end_q]
    correlations = None

    # 1. Correlations
    results['detailed_text_summary'] += "**Correlation Analysis:**\n"
    if target_churn_col not in df.columns:
        errors.append(f"Target churn column '{target_churn_col}' not found.")
        results['detailed_text_summary'] += "- Target column not found.\n"
    else:
        numeric_cols = [target_churn_col] + indicators
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
        if len(df_numeric) > 1:
            correlations = df_numeric.corr(method='pearson')[target_churn_col].drop(target_churn_col).sort_values(key=abs, ascending=False) # Sort by absolute value
            results['correlations'] = correlations
            if not correlations.empty:
                 results['detailed_text_summary'] += f"Pearson correlation coefficients between '{target_churn_col}' and predictors:\n"
                 for ind, val in correlations.items():
                     strength = "Weak"
                     if abs(val) > 0.7: strength = "Strong"
                     elif abs(val) > 0.4: strength = "Moderate"
                     results['detailed_text_summary'] += f"  - {ind}: {val:.3f} ({strength})\n"
            else:
                 results['detailed_text_summary'] += "- No valid correlations could be calculated.\n"
        else:
             errors.append("Not enough valid numeric data for correlation.")
             results['detailed_text_summary'] += "- Not enough valid data points for correlation.\n"

    # 2. Regression
    results['detailed_text_summary'] += "\n**Regression Analysis (Impact Estimation):**\n"
    summary_obj, impact_error = analyze_impact(df, target_churn_col, indicators, date_range)
    if impact_error:
        errors.append(f"Impact Analysis Error: {impact_error}")
        results['detailed_text_summary'] += f"- Error during regression: {impact_error}\n"
    results['regression_summary_obj'] = summary_obj
    if summary_obj:
        try:
            r_squared = float(summary_obj.tables[0].data[0][3])
            adj_r_squared = float(summary_obj.tables[0].data[1][3])
            f_prob = float(summary_obj.tables[0].data[3][3]) # Prob (F-statistic)
            results['detailed_text_summary'] += f"- Model Fit: R-squared={r_squared:.3f}, Adj. R-squared={adj_r_squared:.3f}.\n"
            results['detailed_text_summary'] += f"- Overall Model Significance (Prob F-statistic): {f_prob:.3f} "
            results['detailed_text_summary'] += "(Significant if < 0.05)\n" if f_prob < 0.05 else "(Not significant overall at p=0.05)\n"

            coeffs_df = pd.read_html(summary_obj.tables[1].as_html(), header=0, index_col=0)[0]
            results['detailed_text_summary'] += "- Predictor Coefficients:\n"
            significant_predictors = []
            for predictor, row in coeffs_df.iterrows():
                if predictor == 'const': continue # Skip intercept
                coeff = row['coef']
                p_value = row['P>|t|']
                significance = "Significant (p<0.05)" if p_value < 0.05 else "Not Significant (p>=0.05)"
                results['detailed_text_summary'] += f"  - {predictor}: Coefficient={coeff:.3f}, P-value={p_value:.3f} ({significance})\n"
                if p_value < 0.05:
                    significant_predictors.append(predictor)
            if significant_predictors:
                 results['detailed_text_summary'] += f"- Key Drivers (Significant Predictors): {', '.join(significant_predictors)}\n"
            else:
                 results['detailed_text_summary'] += "- No individual predictors were statistically significant at the p=0.05 level.\n"
        except Exception as e:
            results['detailed_text_summary'] += f"- Could not parse detailed regression summary: {e}\n"
    else:
        results['detailed_text_summary'] += "- Regression analysis could not be completed.\n"

    # 3. Plot
    plot_cols = [target_churn_col] + indicators
    fig, plot_error = plot_trend(df, plot_cols, date_range, title=f"Trend for {target_churn_col} and Predictors")
    if plot_error: errors.append(f"Plotting Error: {plot_error}")
    results['figure'] = fig

    results['errors'] = errors
    return results


# --- 4. Execute Initial Checks and Load/Prepare Data ---
# ... (Keep this section as is) ...
# nltk_messages = check_nltk_resources()
churn_data_raw, macro_data_raw, load_error_message = load_data()
# for msg in nltk_messages: st.toast(msg, icon="ℹ️")
if load_error_message: st.error(load_error_message, icon="🚨"); st.stop()
analysis_data, prep_error_message = prepare_channel_analysis_data(churn_data_raw, macro_data_raw)
if prep_error_message: st.error(prep_error_message, icon="🚨"); st.stop()
if analysis_data is not None:
    churn_channels = [col.replace('Churn_', '') for col in analysis_data.columns if col.startswith('Churn_')]
    macro_indicators_list = [col for col in analysis_data.columns if not col.startswith('Churn_') and col != 'Date']
    available_quarters = sorted(analysis_data['Date'].unique())
else: churn_channels, macro_indicators_list, available_quarters = [], [], []


# --- 5. Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Use the filters above to run an impact analysis, then ask follow-up questions."}]
# analysis_results stores the DICT from the last form submission (incl. detailed summary obj)
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
# detailed_summary_context stores the TEXT summary from the last 'summarize' request or form submission
if "detailed_summary_context" not in st.session_state:
    st.session_state.detailed_summary_context = None

# --- 6. Setup UI ---
# ... (Keep this section as is) ...
st.title("🔬 Analytical Chatbot + LLM")
st.caption("Configure and run impact analysis using the filters below. Then ask follow-up questions about the results.")

# --- Persistent Impact Analysis Configuration Form ---
# ... (Keep this section as is, but update the form submission logic) ...
st.markdown("---")
st.subheader("Impact Analysis Configuration")
with st.form("impact_analysis_form"):
    col_target, col_preds = st.columns([1, 3])
    with col_target: selected_target_channel = st.selectbox("Target Churn Channel:", options=churn_channels, key="target_channel_select", index=0 if churn_channels else None, disabled=not churn_channels)
    with col_preds: selected_indicators = st.multiselect("Predictor Macro Indicator(s):", options=macro_indicators_list, key="indicators_multi_select", disabled=not macro_indicators_list)
    col_start, col_end = st.columns(2)
    with col_start: selected_start_q = st.selectbox("Start Quarter:", options=available_quarters, index=0 if available_quarters else None, key="start_q_select", disabled=not available_quarters)
    with col_end: selected_end_q = st.selectbox("End Quarter:", options=available_quarters, index=len(available_quarters)-1 if available_quarters else None, key="end_q_select", disabled=not available_quarters)
    submitted = st.form_submit_button("📊 Run Impact Analysis", disabled=not analysis_data is not None)

    if submitted:
        if not selected_target_channel: st.warning("Please select a target churn channel.")
        elif not selected_indicators: st.warning("Please select at least one macro indicator.")
        elif not selected_start_q or not selected_end_q: st.warning("Please select a valid date range.")
        elif selected_start_q > selected_end_q: st.warning("Start Quarter cannot be after End Quarter.")
        else:
            with st.spinner("Performing detailed analysis..."):
                target_churn_col = f"Churn_{selected_target_channel}"
                # Store the entire results dictionary
                analysis_results_dict = perform_detailed_impact_analysis(
                    analysis_data, target_churn_col, selected_indicators, selected_start_q, selected_end_q
                )
                st.session_state.analysis_results = analysis_results_dict
                # --- ALSO STORE DETAILED SUMMARY FOR CONTEXT ---
                st.session_state.detailed_summary_context = analysis_results_dict.get('detailed_text_summary', None)

            # Add confirmation message to chat
            st.session_state.messages.append({"role": "assistant", "content": f"Impact analysis complete for {target_churn_col} ({selected_start_q} - {selected_end_q}). Results are displayed below. You can ask me to summarize or explain specific parts."})
            st.rerun() # Rerun to display results and updated chat

# --- Display Analysis Results ---
# Display results from the LATEST form submission
if st.session_state.analysis_results:
    st.markdown("---")
    st.subheader("Latest Impact Analysis Results")
    results = st.session_state.analysis_results

    # Display errors first
    if results.get('errors'):
        for error in results['errors']: st.error(error, icon="🚨")

    # Display Detailed Text Summary (Generated by perform_detailed_impact_analysis)
    if results.get('detailed_text_summary'):
        st.markdown("**Analysis Summary:**")
        st.markdown(results['detailed_text_summary']) # Display the generated detailed summary

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
            st.markdown("**Detailed Regression Stats (Statsmodels):**")
            st.text(results['regression_summary_obj']) # Display full statsmodels summary

    st.markdown("*(Note: Analysis uses dummy data...)*")
    # Keep analysis_results in state until overwritten by next form submission


# --- Chat Interface ---
st.markdown("---")
st.subheader("Ask Questions")

# Display chat messages
container = st.container()
with container:
    # Only display the last N messages to prevent the container from growing too large (optional)
    # display_messages = st.session_state.messages[-10:] # Example: last 10
    display_messages = st.session_state.messages
    for message in display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Handle User Input via chat box ---
if prompt := st.chat_input("Ask follow-up questions about the analysis or general questions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    intent = parse_chat_intent(prompt)
    bot_reply_content = None
    # --- Use the persistent detailed summary context ---
    analysis_context = st.session_state.detailed_summary_context

    # --- Generate Bot Response ---
    with st.spinner("Thinking..."):
        if intent == 'summarize':
             if analysis_context:
                   # Ask Gemini to summarize the detailed summary we already have
                   gemini_prompt = "Re-summarize the key findings from the provided analysis summary, focusing on the main takeaways."
                   bot_reply_content = ask_gemini(gemini_prompt, context=analysis_context, intent='summarize') # Use summarize intent for strict context
             else:
                  bot_reply_content = "No analysis results have been generated yet to summarize. Please run an impact analysis using the filters above first."
        else: # Handles 'explain' and 'unknown'
            # Pass the detailed context (if available) to Gemini.
            # ask_gemini will use it as background for these intents.
            bot_reply_content = ask_gemini(prompt, context=analysis_context, intent=intent)

    # Add assistant response to chat history
    if bot_reply_content:
        st.session_state.messages.append({"role": "assistant", "content": bot_reply_content})

    # Rerun to display the updated chat immediately
    st.rerun()

