from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import load_data, preprocess_data

# Load and preprocess data
df = load_data()  # Load the spam dataset
(X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)  # Preprocess and split the data

# Train Naive Bayes classifier
model = MultinomialNB()  # Initialize the Naive Bayes model
model.fit(X_train, y_train)  # Train the model on the training set

# Evaluate model performance
y_pred = model.predict(X_test)  # Predict on the test set
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score
print(classification_report(y_test, y_pred))  # Print detailed classification metrics

# Save the trained model and vectorizer for future use
joblib.dump(model, "models/spam_classifier.pkl")  # Save the trained model
joblib.dump(vectorizer, "models/vectorizer.pkl")  # Save the vectorizer for text preprocessing

model = joblib.load("models/spam_classifier.pkl")