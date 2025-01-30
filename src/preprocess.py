import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load dataset
def load_data(filename="data/spam.csv"):
    df = pd.read_csv(filename, encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Convert labels to binary (spam=1, ham=0)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# Convert text into numerical features
def preprocess_data(df):
    df['message'] = df['message'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df['message']).toarray()
    y = df['label']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

if __name__ == "__main__":
    df = load_data()
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)
