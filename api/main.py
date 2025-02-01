from fastapi import FastAPI, Request, HTTPException  
from fastapi.responses import HTMLResponse  
from fastapi.templating import Jinja2Templates  
from fastapi.staticfiles import StaticFiles  
from pydantic import BaseModel  
import joblib  
import numpy as np  

# Load the trained model and vectorizer  
model = joblib.load("./model/spam_classifier.joblib")  
vectorizer = joblib.load("./model/vectorizer.joblib")  # Load the vectorizer  

app = FastAPI()  
templates = Jinja2Templates(directory="templates")  
app.mount("/static", StaticFiles(directory="static"), name="static")  

class EmailInput(BaseModel):  
    text: str  

@app.get("/", response_class=HTMLResponse)  
async def read_root(request: Request):  
    return templates.TemplateResponse("index.html", {"request": request})  

@app.post("/predict/")  
async def predict_spam(email: EmailInput):  
    try:  
        # Preprocess the input text and vectorize it  
        preprocessed_text = preprocess_text(email.text)  # Preprocess the input  
        email_vectorized = vectorizer.transform([preprocessed_text])  # Vectorize the preprocessed text  
        
        # Predict using the model  
        prediction = model.predict(email_vectorized)[0]  # Predict  
        
        # Return the result based on the model output  
        return {  
            "prediction": "This message is Spam, Be aware" if prediction == 1 else "This message is not Spam, You can read it"  
        }  
    
    except Exception as e:  
        # Log the exception and return an error response  
        raise HTTPException(status_code=400, detail=f"An error occurred during prediction: {str(e)}")  

def preprocess_text(text: str):  
    """  
    Preprocess the input text to match the format used during training.  
    """  
    text = text.lower()  # Convert to lowercase  
    # You can add more preprocessing steps here if necessary (e.g., removing punctuation, stemming, etc.)  
    return text  # Return the preprocessed text