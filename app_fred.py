# app.py
import streamlit as st
import data_fetcher  # Your existing module for fetching data and definitions
import nltk
import random
import os

# --- NLTK Download Check ---
# Streamlit apps might need this check on first run or deployment
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    st.write("Downloading NLTK 'punkt' resource...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    st.write("Downloading NLTK 'stopwords' resource...")
    nltk.download('stopwords', quiet=True)
# Add other nltk downloads if needed (e.g., 'averaged_perceptron_tagger')

# --- Gemini LLM Setup ---
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        # Try getting it from Streamlit secrets if deployed
        try:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        except:
             st.warning("GEMINI_API_KEY not found in environment variables or Streamlit secrets.", icon="⚠️")
             genai_configured = False
    else:
        genai_configured = True # Assume configured if key exists

    if genai_configured:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a suitable model
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # st.success("Gemini LLM Initialized.", icon="✅") # Optional confirmation
    else:
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


# --- Chatbot Logic (adapted from chatbot.py) ---

# Keywords (can be refined)
GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
FAREWELL_KEYWORDS = ("bye", "goodbye", "quit", "exit", "see ya")
DEFINITION_KEYWORDS = ("define", "definition", "what is")
DATA_KEYWORDS = ("latest", "value", "number", "figure", "data", "rate", "level", "how much", "what's the")
EXPLAIN_KEYWORDS = ("explain", "tell me more about", "why is", "significance of")

# Responses
GREETING_RESPONSES = ["Hello!", "Hi there!", "Greetings!", "How can I help you with macroeconomic data today?"]
UNKNOWN_RESPONSES = [
    "Sorry, I didn't quite understand that. Could you rephrase?",
    "I can provide the 'latest value' or 'definition' for indicators like Real GDP, CPI, Unemployment Rate, Fed Funds Rate, etc. You can also ask me to 'explain' an indicator.",
    "My apologies, I'm focused on macroeconomic indicators right now."
]
LLM_ERROR_RESPONSE = "Sorry, I encountered an issue trying to access my advanced knowledge module."


def ask_gemini(prompt):
    """Sends a prompt to the configured Gemini model."""
    if not genai_configured or not gemini_model:
        return "My advanced knowledge module (Gemini LLM) is not configured or available."
    st.info("Asking Gemini...", icon="🧠") # Indicate LLM call in UI
    try:
        response = gemini_model.generate_content(prompt)
        if response.parts:
            return response.text
        else:
            st.warning(f"Gemini response issue: {response.prompt_feedback}", icon="⚠️")
            return "I received an unusual response from my advanced module. Please try rephrasing."
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}", icon="🚨")
        return LLM_ERROR_RESPONSE


def get_bot_response(user_input):
    """Processes user input and returns the bot's response."""
    user_input_lower = user_input.lower()
    tokens = nltk.word_tokenize(user_input_lower)

    # Simple check for greetings first
    if any(word in tokens for word in GREETING_KEYWORDS):
        return random.choice(GREETING_RESPONSES)

    # (We don't need farewell keywords to exit the app like in the CLI)

    intent = None
    entity = None
    explain_intent = any(word in user_input_lower for word in EXPLAIN_KEYWORDS)

    matched_keywords = sorted(data_fetcher.indicator_map.keys(), key=len, reverse=True)
    for indicator_keyword in matched_keywords:
        if indicator_keyword in user_input_lower:
            entity = indicator_keyword
            break

    if not explain_intent:
        if any(word in tokens for word in DEFINITION_KEYWORDS):
            intent = "get_definition"
        elif any(word in tokens for word in DATA_KEYWORDS):
            intent = "get_data"

    # --- Rule-Based Response Generation ---
    response = None

    if explain_intent and entity:
        with st.spinner(f"Gathering info to explain {entity}..."): # Show spinner
            definition = data_fetcher.get_definition(entity)
            data_details = data_fetcher.indicator_map.get(entity)
            latest_data = "N/A"
            if data_details and data_details["source"] == "fred":
                latest_data = data_fetcher.get_fred_series_latest(data_details['id'])

            prompt = f"Explain the macroeconomic indicator '{entity}'.\n"
            if not "Sorry, I don't have a definition" in definition:
                prompt += f"Definition: {definition}\n"
            if latest_data != "N/A" and "Error" not in latest_data and "not initialized" not in latest_data:
                prompt += f"The latest value is: {latest_data}.\n"
            prompt += f"Focus on its significance and what its current value might indicate. Keep the explanation concise but informative, suitable for a chatbot response.\nOriginal question was: '{user_input}'"
            response = ask_gemini(prompt)

    elif intent == "get_definition" and entity:
        response = data_fetcher.get_definition(entity)

    elif intent == "get_data" and entity:
        indicator_details = data_fetcher.indicator_map.get(entity)
        if indicator_details:
            desc = indicator_details.get('desc', entity)
            with st.spinner(f"Fetching latest {desc}..."): # Show spinner
                if indicator_details["source"] == "fred":
                    data_value = data_fetcher.get_fred_series_latest(indicator_details['id'])
                    response = f"The latest {desc} is: {data_value}"
                elif indicator_details["source"] == "worldbank":
                    country = indicator_details.get('country', 'US')
                    data_value = data_fetcher.get_world_bank_indicator_latest(indicator_details['id'], country)
                    response = f"The latest {desc} (World Bank, {country}) is: {data_value}"
                else:
                    response = f"Sorry, I don't know how to fetch data for {entity} from '{indicator_details['source']}' yet."
        else:
            response = f"Sorry, I recognized '{entity}' but couldn't find how to fetch its data."

    # --- Fallback Logic ---
    if response is None:
        if entity and not intent and not explain_intent:
            response = f"What would you like to know about {entity}? Try 'latest value', 'definition', or 'explain'."
        elif intent and not entity:
            response = "Which indicator are you interested in? (e.g., Real GDP, CPI, Fed Funds Rate)"
        else:
            # Fallback to Gemini for general queries
            prompt = f"As a macroeconomic chatbot assistant, answer the following user query concisely: '{user_input}'"
            response = ask_gemini(prompt)
            if response == LLM_ERROR_RESPONSE or "My advanced knowledge module" in response:
                 response = random.choice(UNKNOWN_RESPONSES)

    return response

# --- Streamlit App Interface ---

st.set_page_config(page_title="Macroeconomic Chatbot", page_icon="📈")
st.title("📈 Macroeconomic Indicator Chatbot")
st.caption("Ask about US indicators like GDP, CPI, Unemployment, Fed Funds Rate, or ask for explanations!")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial welcome message from assistant
    st.session_state.messages.append({"role": "assistant", "content": random.choice(GREETING_RESPONSES)})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # Use markdown for better formatting

# React to user input
if prompt := st.chat_input("Ask about indicators (e.g., 'latest real gdp', 'define inflation', 'explain trade balance')"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    bot_reply = get_bot_response(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# --- Optional: Add a sidebar for info or settings ---
# with st.sidebar:
#     st.header("About")
#     st.write("This chatbot provides information on key US macroeconomic indicators using data from FRED and explanations via Google Gemini.")
#     st.write("Ensure FRED_API_KEY and GEMINI_API_KEY are set as environment variables or Streamlit secrets.")