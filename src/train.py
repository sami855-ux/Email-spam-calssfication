from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import load_data, preprocess_data

# Load and preprocess data
df = load_data()
(X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
