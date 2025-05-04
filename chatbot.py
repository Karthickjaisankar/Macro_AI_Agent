# chatbot.py
import data_fetcher # Our updated data functions
import nltk
import random
import os # To get environment variables

# --- Gemini LLM Setup ---
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not found.")
        genai_configured = False
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a suitable model, like gemini-1.5-flash for speed and cost-effectiveness
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        genai_configured = True
        print("Gemini LLM Initialized.") # Confirmation
except ImportError:
    print("Warning: google.generativeai library not found. LLM features disabled.")
    print("Install it using: pip install google-generativeai")
    genai_configured = False
except Exception as e:
    print(f"Error initializing Gemini API: {e}")
    genai_configured = False
# --- End Gemini Setup ---


# Basic keywords (mostly unchanged, added 'explain')
GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
FAREWELL_KEYWORDS = ("bye", "goodbye", "quit", "exit", "see ya")
DEFINITION_KEYWORDS = ("define", "definition", "what is")
DATA_KEYWORDS = ("latest", "value", "number", "figure", "data", "rate", "level", "how much", "what's the")
EXPLAIN_KEYWORDS = ("explain", "tell me more about", "why is", "significance of")

# Responses (unchanged)
GREETING_RESPONSES = ["Hello!", "Hi there!", "Greetings!", "How can I help you with macroeconomic data today?"]
FAREWELL_RESPONSES = ["Goodbye!", "See you later!", "Have a great day!"]
# Refined Unknown Responses
UNKNOWN_RESPONSES = [
    "Sorry, I didn't quite understand that. Could you rephrase?",
    "I can provide the 'latest value' or 'definition' for indicators like Real GDP, CPI, Unemployment Rate, Fed Funds Rate, etc. You can also ask me to 'explain' an indicator.",
    "My apologies, I'm focused on macroeconomic indicators right now."
    ]
LLM_ERROR_RESPONSE = "Sorry, I encountered an issue trying to access my advanced knowledge module."


# --- Gemini Interaction Function ---
def ask_gemini(prompt):
    """Sends a prompt to the configured Gemini model and returns the text response."""
    if not genai_configured:
        return "My advanced knowledge module (Gemini LLM) is not configured or available."
    print(">>> Asking Gemini...") # Indicate LLM call
    try:
        response = gemini_model.generate_content(prompt)
        # Basic check if response has text (might need more robust error checking)
        if response.parts:
             return response.text
        else:
             # Handle cases where the response might be blocked or empty
             print(f"Gemini response issue: {response.prompt_feedback}")
             return "I received an unusual response from my advanced module. Please try rephrasing."
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Consider specific error handling (e.g., rate limits, auth errors)
        return LLM_ERROR_RESPONSE


def process_input(user_input):
    """
    Analyzes user input, tries rule-based response, falls back to LLM if needed.
    """
    user_input_lower = user_input.lower()
    tokens = nltk.word_tokenize(user_input_lower)

    # 1. Check for Greetings/Farewells (same as before)
    if any(word in tokens for word in GREETING_KEYWORDS):
        return random.choice(GREETING_RESPONSES)
    if any(word in tokens for word in FAREWELL_KEYWORDS):
        return random.choice(FAREWELL_RESPONSES)

    # 2. Identify Potential Intent & Entity
    intent = None
    entity = None
    explain_intent = any(word in user_input_lower for word in EXPLAIN_KEYWORDS) # Separate check for explain

    # Find the indicator (entity) - check against our known indicators
    # Prioritize longer matches first if keywords overlap (e.g., "fed funds rate" vs "rate")
    matched_keywords = sorted(data_fetcher.indicator_map.keys(), key=len, reverse=True)
    for indicator_keyword in matched_keywords:
        if indicator_keyword in user_input_lower:
            entity = indicator_keyword
            break # Take the first (longest) match found

    # Determine primary intent (Data or Definition) only if not an explain intent
    if not explain_intent:
        if any(word in tokens for word in DEFINITION_KEYWORDS):
            intent = "get_definition"
        elif any(word in tokens for word in DATA_KEYWORDS):
            intent = "get_data"

    # --- Rule-Based Response Generation ---
    response = None # Initialize response

    if explain_intent and entity:
        # Handle explanation requests
        definition = data_fetcher.get_definition(entity)
        data_details = data_fetcher.indicator_map.get(entity)
        latest_data = "N/A"
        if data_details and data_details["source"] == "fred": # Only fetch data if source is known
             latest_data = data_fetcher.get_fred_series_latest(data_details['id'])

        # Formulate a prompt for Gemini
        prompt = f"Explain the macroeconomic indicator '{entity}'.\n"
        if not "Sorry, I don't have a definition" in definition:
             prompt += f"Definition: {definition}\n"
        if latest_data != "N/A" and "Error" not in latest_data:
             prompt += f"The latest value is: {latest_data}.\n"
        prompt += f"Focus on its significance and what its current value might indicate. Keep the explanation concise but informative, suitable for a chatbot response.\nOriginal question was: '{user_input}'"
        response = ask_gemini(prompt)

    elif intent == "get_definition" and entity:
        response = data_fetcher.get_definition(entity)

    elif intent == "get_data" and entity:
        indicator_details = data_fetcher.indicator_map.get(entity)
        if indicator_details:
            desc = indicator_details.get('desc', entity) # Get description for clarity
            if indicator_details["source"] == "fred":
                data_value = data_fetcher.get_fred_series_latest(indicator_details['id'])
                response = f"The latest {desc} is: {data_value}"
            elif indicator_details["source"] == "worldbank":
                # Add country handling if needed later
                country = indicator_details.get('country', 'US')
                data_value = data_fetcher.get_world_bank_indicator_latest(indicator_details['id'], country)
                response = f"The latest {desc} (World Bank, {country}) is: {data_value}"
            else:
                response = f"Sorry, I don't know how to fetch data for {entity} from '{indicator_details['source']}' yet."
        else:
             response = f"Sorry, I recognized '{entity}' but couldn't find how to fetch its data."

    # --- Fallback Logic ---
    if response is None: # If no rule-based response was generated
        if entity and not intent and not explain_intent:
             # User mentioned indicator but not what they want specifically
             response = f"What would you like to know about {entity}? Try 'latest value', 'definition', or 'explain'."
        elif intent and not entity:
             # User asked for data/definition but not which indicator
             response = "Which indicator are you interested in? (e.g., Real GDP, CPI, Fed Funds Rate)"
        else:
             # Could not determine intent or entity clearly, or general question -> Ask Gemini
             print(">>> Falling back to Gemini for general query...")
             prompt = f"As a macroeconomic chatbot assistant, answer the following user query concisely: '{user_input}'"
             response = ask_gemini(prompt)
             # If Gemini also fails, provide a standard unknown response
             if response == LLM_ERROR_RESPONSE or "My advanced knowledge module" in response:
                  response = random.choice(UNKNOWN_RESPONSES)

    return response


# --- Main Chat Loop (Unchanged) ---
if __name__ == "__main__":
    print(random.choice(GREETING_RESPONSES) + " (Type 'bye' to exit)")
    while True:
        user_text = input("You: ")
        bot_response = process_input(user_text)
        print(f"Bot: {bot_response}")

        # Check if the response is a farewell to break the loop
        if bot_response in FAREWELL_RESPONSES:
            break