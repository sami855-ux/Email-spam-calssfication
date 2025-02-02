import pandas as pd  
import re  
import string  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.preprocessing import LabelEncoder  

# Load dataset from a CSV file  
def load_data(filename):  
    """  
    Load and preprocess the dataset.  

    Parameters:  
    filename (str): Path to the CSV file.  

    Returns:  
    pd.DataFrame: Processed DataFrame with 'label' and 'message' columns.  
    """  
    # Read the dataset (some datasets may require 'latin-1' encoding to avoid decoding errors)  
    df = pd.read_csv(filename, encoding="latin-1")  

    # Select only the relevant columns: 'v1' (label) and 'v2' (message)  
    df = df[['v1', 'v2']]  
    
    # Rename columns for clarity  
    df.columns = ['label', 'message']  
    
    # Handle missing values in the 'label' and 'message' columns  
    # Remove rows with missing values in critical columns  
    df.dropna(subset=['label', 'message'], inplace=True)  

    # Convert labels to binary (ham=0, spam=1)  
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  

    # Check if any labels are missing after mapping  
    if df['label'].isnull().any():  
        raise ValueError("After mapping, some labels are missing. Please check the data.")  

    return df  

# Clean text messages  
def clean_text(text):  
    """  
    Perform basic text preprocessing.  

    Returns:  
    str: Cleaned text.  
    """  
    text = text.lower()  # Convert text to lowercase  
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation  
    text = re.sub(r"\d+", "", text)  # Remove numbers  
    return text  

# Convert text messages into numerical features  
def preprocess_data(df):  
    """  
    Preprocess text data by cleaning and vectorizing.  

    Parameters:  
    df (pd.DataFrame): DataFrame containing 'message' and 'label'.  

    Returns:  
    tuple: Train-test split data (X_train, X_test, y_train, y_test) and vectorizer.  
    """  
    # Apply text cleaning function to all messages  
    df['message'] = df['message'].apply(clean_text)  

    # Check for outliers in the number of characters per message  
    message_length = df['message'].str.len()  
    # Define thresholds for outliers if needed (e.g., messages longer than 1000 characters)  
    outlier_threshold = 1000  
    outliers = df[message_length > outlier_threshold]  # Get outliers if any  
    if not outliers.empty:  
        print(f"Removing {len(outliers)} outlier messages longer than {outlier_threshold} characters.")  
        df = df[message_length <= outlier_threshold]  # Remove outliers  

    # Initialize TF-IDF Vectorizer with a max feature limit to reduce dimensionality  
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  

    # Transform messages into numerical feature vectors  
    X = vectorizer.fit_transform(df['message']).toarray()  
    
    # Extract labels  
    y = df['label']  

    # Split data into training and testing sets (80% train, 20% test)  
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer  

if __name__ == "__main__":  
    # Load the dataset  
    df = load_data("./data/spam.csv")  

    # Preprocess data and obtain train-test split and vectorizer  
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data(df)