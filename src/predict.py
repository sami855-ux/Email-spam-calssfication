import joblib
import numpy as np

# Load trained model and vectorizer
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_email(text):
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    return "Spam" if prediction == 1 else "Ham"

if __name__ == "__main__":
    email_text = input("Enter email text: ")
    print("Prediction:", predict_email(email_text))
