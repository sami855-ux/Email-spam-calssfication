from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.model_selection import GridSearchCV  
from sklearn.dummy import DummyClassifier  # Import DummyClassifier for baseline  
import joblib  
from preprocess import load_data, preprocess_data  
import matplotlib.pyplot as plt  # Import matplotlib for plotting  
import seaborn as sns  # Import seaborn for enhanced visuals  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For confusion matrix visualization  

# Load and preprocess data  
df = load_data("./data/spam.csv")  
(X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)  

# Initialize the Naive Bayes model  
model = MultinomialNB()  

# Define the hyperparameter grid to search  
param_grid = {  
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  
    'fit_prior': [True, False]            
}  

# Set up the GridSearchCV  
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,   
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=1)  

# Fit GridSearchCV to the training data  
grid_search.fit(X_train, y_train)  

# Best parameters from the grid search  
print("Best Hyperparameters:", grid_search.best_params_)  

# Make predictions using the best estimator  
best_model = grid_search.best_estimator_  
y_pred = best_model.predict(X_test)  

# Evaluate model performance  
print("Accuracy:", accuracy_score(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

# Create and fit the DummyClassifier for baseline comparison  
dummy_clf = DummyClassifier(strategy="most_frequent")  # Can adjust strategy as needed  
dummy_clf.fit(X_train, y_train)  
y_dummy_pred = dummy_clf.predict(X_test)  

# Evaluate Dummy Classifier  
dummy_accuracy = accuracy_score(y_test, y_dummy_pred)  
print("Dummy Classifier Accuracy:", dummy_accuracy)  

# Plotting accuracy comparison  
accuracy_scores = {  
    'Naive Bayes': accuracy_score(y_test, y_pred),  
    'Dummy Classifier': dummy_accuracy  
}  

plt.figure(figsize=(8, 5))  
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))  
plt.title('Model Accuracy Comparison')  
plt.ylabel('Accuracy')  
plt.ylim(0, 1)  
plt.show()  

# Confusion Matrix for Naive Bayes  
cm = confusion_matrix(y_test, y_pred)  
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)  
cm_display.plot(cmap=plt.cm.Blues)  
plt.title('Confusion Matrix for Naive Bayes Classifier')  
plt.show()  

# Confusion Matrix for Dummy Classifier  
dummy_cm = confusion_matrix(y_test, y_dummy_pred)  
dummy_cm_display = ConfusionMatrixDisplay(confusion_matrix=dummy_cm, display_labels=dummy_clf.classes_)  
dummy_cm_display.plot(cmap=plt.cm.Blues)  
plt.title('Confusion Matrix for Dummy Classifier')  
plt.show()  

# Save the trained model and vectorizer for future use  
joblib.dump(best_model, "./model/spam_classifier.joblib")  
joblib.dump(vectorizer, "./model/vectorizer.joblib")