from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load("spam_classifier.pkl")

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
    # Preprocess the input text manually (e.g., tokenization, feature extraction)
    # This step depends on how the model was trained
    preprocessed_text = preprocess_text(email.text)  # Replace with your preprocessing logic

    # Predict using the model
    prediction = model.predict([preprocessed_text])[0]  # Ensure input is in the correct format
    return {"prediction": "This message is Spam, Be aware" if prediction == 1 else "This message is not Spam, You can read it"}

def preprocess_text(text: str):
    """
    Preprocess the input text to match the format used during training.
    Replace this with your actual preprocessing logic.
    """
    # Example: Convert text to lowercase and extract features
    text = text.lower()  # Convert to lowercase
    # Add more preprocessing steps as needed (e.g., tokenization, removing stopwords, etc.)
    return text  # Return preprocessed text (this should match the format expected by the model)