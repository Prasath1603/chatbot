import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def predict_sentiment(text):
    text = clean_text(text)
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)
    return prediction[0]

def generate_response(sentiment):
    responses = {
        "happy": "That's great to hear! Keep it up!",
        "sad": "I'm here for you. It's okay to feel down sometimes.",
        "angry": "Try to take deep breaths. Want to talk more about it?",
        "neutral": "Thanks for sharing. Want to tell me more?"
    }
    return responses.get(sentiment, "Thanks for sharing. Let's talk more.")