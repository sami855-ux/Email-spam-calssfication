from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("../models/spam_classifier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

app = FastAPI()

class EmailInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_spam(email: EmailInput):
    transformed_text = vectorizer.transform([email.text])
    prediction = model.predict(transformed_text)[0]
    return {"prediction": "Spam" if prediction == 1 else "Ham"}

# Run with: uvicorn main:app --reload
