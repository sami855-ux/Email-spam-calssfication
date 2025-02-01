import joblib
import numpy as np

# Load trained model and vectorizer
model = joblib.load("./model/spam_classifier.joblib")
vectorizer = joblib.load("./model/vectorizer.joblib")

def predict_email(text):
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    return "Spam" if prediction == 1 else "Ham"

if __name__ == "__main__":
    email_text = input("Enter email text: ")
    print("Prediction:", predict_email(email_text))
    
# import joblib
# import numpy as np

# # Load the trained spam classifier model and vectorizer from disk
# model = joblib.load("models/spam_classifier.pkl")

# def predict_email(text):
#     """
#     Predict whether an email is spam or not.

#     Parameters:
#     text (str): The email message text.

#     Returns:
#     str: "Spam" if classified as spam, otherwise "Ham" (not spam).
#     """
#     # Transform the input text using the loaded vectorizer
#     text = vectorizer.transform([text])

#     # Use the trained model to predict whether it's spam (1) or ham (0)
#     prediction = model.predict(text)[0]

#     # Return the corresponding label based on the prediction
#     return "Spam" if prediction == 1 else "Ham"

# if __name__ == "__main__":
#     # Prompt the user to enter email text for classification
#     email_text = input("Enter email text: ")

#     # Print the
#     print("Prediction:", predict_email(email_text))