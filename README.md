# Building a Spam Classifier with FastAPI

In the digital age, spam emails pose considerable challenges to users, necessitating robust solutions for effective management. This essay outlines the development of a spam classification web application using FastAPI, a modern Python web framework, coupled with pre-trained machine learning models and vectorization techniques.

## Problem Definition and Data Acquisition

The goal of spam detection is to automatically classify incoming emails as either spam (unwanted or harmful) or ham (legitimate communication). Implementing an effective spam detection system helps improve user experience, reduces phishing risks, and enhances the overall quality of email communications.

I used dataset in csv file structure, This dataset is used for training models to classify emails as "spam" or "ham" (non-spam). The effectiveness of the model depends on its ability to accurately distinguish between the two categories.

- DatasetName : spam.csv
- Source: Kaggle
- Link to Dataset: https://kaggle.com
- Retrieval Date: 21/05/2017E.c


The dataset can be used with algorithms such as Naive Bayes, Support Vector Machines (SVM), or neural networks to automate the detection of spam emails. But i used Navie Bayes in this project. 

 ### Data understanding and exploration is found at the dirctory of notebooks folder

# Machine learning algorithm used in this project
Since spam detection involves categorizing emails into distinct classes (spam vs. ham), it employs classification algorithms like Naive Bayes, SVM, or logistic regression (note: logistic regression is a classification technique despite its name).

Every dependencies that is used in this project are placed at requirements.txt

# Application Structure

This FastAPI application consists of several key components:
## Train the dataset

Model and Vectorizer Loading: The application begins by loading the trained spam classification model and the vectorizer using the Joblib library. This setup allows for swift execution of predictions on new email inputs.

```python
# Load and preprocess data  
df = load_data("./data/spam.csv")  # Load the spam dataset  
(X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df) # Preprocess and split the data  

# Train Naive Bayes classifier  
model = MultinomialNB()  # Initialize the Naive Bayes model  
model.fit(X_train, y_train)  # Train the model on the training set  

# Evaluate model performance  
y_pred = model.predict(X_test)  # Predict on the test set  
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score  
print(classification_report(y_test, y_pred))  # Print detailed classification metrics

model = joblib.dump("./model/spam_classifier.joblib")  
vectorizer = joblib.dump("./model/vectorizer.joblib")
```

To visualize it run python src/train.py

## Predict using terminal
Load the joblib files 

 ```python
model = joblib.load("./model/spam_classifier.joblib")
vectorizer = joblib.load("./model/vectorizer.joblib")
 ```

To visualize it run python src/predict.py

## Web Application Setup

An instance of the FastAPI application is created, while templates and static files are configured to enable a rich user interface that displays the prediction results.

```python
app = FastAPI()  
templates = Jinja2Templates(directory="templates")  
app.mount("/static", StaticFiles(directory="static"), name="static")
```
## Input Model Definition

A Pydantic model named EmailInput is defined to structure the incoming data. This model validates that the input is a string containing the email text.

```python
class EmailInput(BaseModel):  
 text: str
```
Routing and Logic
The application implements two main routes:

The root ("/") route renders the home page with an HTML template.

```python
@app.get("/", response_class=HTMLResponse)  
async def read_root(request: Request):  
 return templates.TemplateResponse("index.html", {"request": request})
```
The prediction route ("/predict/") processes the input email, applying preprocessing and utilizing the loaded model for classification.

```python
@app.post("/predict/")  
async def predict_spam(email: EmailInput):
```
## Prediction logic

Prediction Logic
The prediction process involves several steps:

```python
def predict_email(text):
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    return "Spam" if prediction == 1 else "Ham"
```

## Text Preprocessing

The input text is transformed to lower case. Additional preprocessing, such as removing punctuation, can be added as needed.

```python
def preprocess_text(text: str):  
 return text.lower()
```

Vectorization: The preprocessed email text is vectorized using the previously loaded vectorizer.

Model Prediction: The model predicts whether the email is spam or not. The output is appropriately formatted and returned in the response.

## Error Handling

To maintain robustness, the application incorporates error handling using FastAPI’s HTTPException. This ensures that any unexpected issues during the prediction process are logged and a user-friendly message is returned.

```python
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
```

## Model selection training details

#### Naive Bayes
The reason i used this model is that it is straightforward to implement and efficient in terms of computation. It works well with large datasets, and  It’s particularly suited for text classification tasks (like spam detection) because it assumes that the presence of a feature (word) in a class is independent of the presence of any other feature.

```python
model = MultinomialNB()  # Initialize the Naive Bayes model  
model.fit(X_train, y_train)  # Train the model on the training set
```

## Model Evaluation metrics

```python
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score  
print(classification_report(y_test, y_pred))  # Print detailed classification report
```
The accuracy score is calculated using the accuracy_score function by comparing y_test (true labels) with y_pred (predicted labels). It shows the overall proportion of correctly classified emails.

The classification_report function provides a comprehensive overview of the model's performance, reporting precision, recall, F1-score, and support for each class (spam and ham). This detailed report gives insights into how well the model is performing and helps identify areas for improvement.

# Steps for Deployment on Render

- step 1: Create a Web Application
- step 2: Create requirements.txt -> This file contains all the libraries my application requires.
- step 3: Version Control with Git -> creating reppository using github and git
- step 4: Create a Render Account
- step 5: Connect Your Repository -> Link my GitHub account and select the repository i created for my spam detection application. Render will automatically deploy the web application each time i push changes to this repository.
- step 6: Configure Your Render Service -> 
Service Name: My service a descriptive name.
Environment: Choose Python.
Build Command: pip install -r requirements.txt.
Start Command: uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload. the use of api is that main.py is found at the folder of api


## Conclusion

This FastAPI application for spam classification illustrates an efficient integration of web technologies with machine learning models. By following a simple structure that includes model loading, a user-friendly interface, and robust error handling, developers can swiftly reproduce and deploy similar projects in the future. With the growing risks of spam in modern communication, such systems are essential to enhance user experiences and improve email management strategies.
