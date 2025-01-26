from flask import Flask, request, jsonify, render_template
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
from fuzzywuzzy import process
import nltk

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Explicitly pass the API key for testing
GEMINI_API_KEY = "AIzaSyCaUIMJ54yhOWnz_i8XcxCOX47ZvzQZhBw"  # Replace this with your actual key

# Flask app initialization
app = Flask(__name__)

# Chatbot 1 Functions
# Load dataset for Chatbot 1
def load_dataset_chatbot1(path):
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        df.fillna('', inplace=True)
        df['Searchable'] = (
            df['Name'].astype(str) + ' ' +
            df['Mood'].astype(str) + ' ' +
            df['Activity'].astype(str) + ' ' +
            df['Tempo'].astype(str) + ' ' +
            df['Language'].astype(str)
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{path}' does not exist. Please check the path and try again.")
    except Exception as e:
        raise RuntimeError(f"Error loading the file: {e}")

# Configure Generative AI for Chatbot 1
def configure_api_chatbot1():
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # Use the API key directly here
    except Exception as e:
        raise RuntimeError(f"Error configuring Generative AI API: {e}")

# Chatbot 1 Response Generation
def chatbot_response_chatbot1(user_input):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = (
            f"You are a highly knowledgeable and adaptive chatbot specializing in music recommendations. "
            f"Provide accurate and verified song recommendations based on user preferences. If specific details like "
            f"song names, artists, or movie names are unavailable, respond politely and avoid guessing. Also, respond "
            f"to general user queries thoughtfully and appropriately. "
            f"User Input: {user_input}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I'm sorry, I encountered an error while processing your input: {str(e)}"

# Music Recommendation for Chatbot 1
def recommend_music_chatbot1(df, user_query):
    try:
        choices = df['Searchable'].tolist()
        matches = process.extract(user_query, choices, limit=5)
        top_matches = [match for match in matches if match[1] > 60]  # Filter based on confidence score

        recommendations = df.iloc[[choices.index(match[0]) for match in top_matches]][['Name', 'Artist']]
        return recommendations.to_dict('records')
    except Exception as e:
        return []

# Chatbot 2 Functions
# Load dataset for Chatbot 2
def load_dataset_chatbot2(path):
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        df.fillna('', inplace=True)
        df['Searchable'] = (
            df['Name'].astype(str) + ' ' +
            df['Mood'].astype(str) + ' ' +
            df['Activity'].astype(str) + ' ' +
            df['Tempo'].astype(str) + ' ' +
            df['Language'].astype(str)
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{path}' does not exist. Please check the path and try again.")
    except Exception as e:
        raise RuntimeError(f"Error loading the file: {e}")

# Configure Generative AI for Chatbot 2
def configure_api_chatbot2():
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # Use the API key directly here
    except Exception as e:
        raise RuntimeError(f"Error configuring Gemini API: {e}")

# Chatbot 2 Response Generation
def chatbot_response_chatbot2(user_input, mood_context=None):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = (
            f"You are a dynamic, empathetic, and creative chatbot who provides emotional support "
            f"and personalized music recommendations. Respond uniquely to user inputs, and adapt "
            f"to their mood and context. Current Mood Context: {mood_context}. User Input: {user_input}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Music Recommendation for Chatbot 2
def recommend_music_chatbot2(df, user_query):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['Searchable'])
        query_vec = vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        recommendations = df.iloc[top_indices][['Name', 'Artist']]
        return recommendations.to_dict('records')
    except Exception as e:
        return []

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in a 'templates' folder

@app.route('/chatbot1', methods=['GET'])
def chatbot1_page():
    return render_template('chatbot1.html')  # Render chatbot1 page

@app.route('/chatbot2', methods=['GET'])
def chatbot2_page():
    return render_template('chatbot2.html')

@app.route('/chatbot1', methods=['POST'])
def chatbot1():
    dataset_path = r"C:\Users\Admin\crescendo\music2.xlsx"

    try:
        df = load_dataset_chatbot1(dataset_path)
        configure_api_chatbot1()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    recommendations = recommend_music_chatbot1(df, user_query)
    if recommendations:
        return jsonify({"recommendations": recommendations})
    else:
        ai_response = chatbot_response_chatbot1(user_query)
        return jsonify({"ai_response": ai_response})

@app.route('/chatbot2', methods=['POST'])
def chatbot2():
    dataset_path = r"C:\Users\Admin\crescendo\music2.xlsx"

    try:
        df = load_dataset_chatbot2(dataset_path)
        configure_api_chatbot2()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    user_query = request.json.get("message", "")
    mood_context = request.json.get("mood", "")

    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    if not mood_context:
        mood_context = "neutral"  # If no mood context is provided, use a default value

    ai_response = chatbot_response_chatbot2(user_query, mood_context)

    return jsonify({"response": ai_response})


if __name__ == "__main__":
    app.run(debug=True)
