from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("C:/Users/samit/OneDrive/Desktop/ml22/Email-spam-calssfication/models/spam_classifier.pkl")
vectorizer = joblib.load("C:/Users/samit/OneDrive/Desktop/ml22/Email-spam-calssfication/models/vectorizer.pkl")

app = FastAPI()

class EmailInput(BaseModel):
    text: str

@app.get("/")  
def read_root():  
    return {"message": "Model loaded successfully!"} 

@app.post("/predict/")
def predict_spam(email: EmailInput):
    transformed_text = vectorizer.transform([email.text])
    prediction = model.predict(transformed_text)[0]
    return {"prediction": "This message is Spam, Be aware" if prediction == 1 else "This message is not Spam, You can read it"}

# Run with: uvicorn api.main:app --reload
