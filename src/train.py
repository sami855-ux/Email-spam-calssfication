from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV  
import joblib  
from preprocess import load_data, preprocess_data  

# Load and preprocess data  
df = load_data("./data/spam.csv")  # Load the spam dataset  
(X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)  # Preprocess and split the data  

# Initialize the Naive Bayes model  
model = MultinomialNB()  

# Define the hyperparameter grid to search  
param_grid = {  
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # Smoothing parameter  
    'fit_prior': [True, False]            # Whether to learn class prior probabilities  
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
y_pred = best_model.predict(X_test)  # Predict on the test set  

# Evaluate model performance  
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score  
print(classification_report(y_test, y_pred))  # Print detailed classification metrics  

# Save the trained model and vectorizer for future use  
joblib.dump(best_model, "./model/spam_classifier.joblib")  # Save the best model in Joblib format  
joblib.dump(vectorizer, "./model/vectorizer.joblib")  # Save the vectorizer