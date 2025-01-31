from fastapi import FastAPI, Request  
from fastapi.responses import HTMLResponse  
from fastapi.templating import Jinja2Templates  
from fastapi.staticfiles import StaticFiles 
from pydantic import BaseModel  
import joblib  

# Load model and vectorizer  
model = joblib.load("C:/Users/samit/OneDrive/Desktop/ml22/Email-spam-calssfication/models/spam_classifier.pkl")  
vectorizer = joblib.load("C:/Users/samit/OneDrive/Desktop/ml22/Email-spam-calssfication/models/vectorizer.pkl")  

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
    transformed_text = vectorizer.transform([email.text])  
    prediction = model.predict(transformed_text)[0]  
    return {"prediction": "This message is Spam, Be aware" if prediction == 1 else "This message is not Spam, You can read it"}