from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import joblib

import spacy
import re

custom_stop_words = ["hotel", "room", "day", "night", "time", "got",
                    "people", "asked", "told", "thing", "really", "want",
                    "go", "stay", "come", "look", "say", "try", "think",
                    "service", "nice", "place", "know", "little", "check", 
                    "small", "bit", "big", "lot", "n't", "way", "close", "work", "need"]


# Load spaCy's English model for lemmatization
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

batch_size = 1

def preprocess_text(df, text_col="Review", custom_stop_words=custom_stop_words):
    """
    Function to clean the text and assign clean text into a new column using the nlp pipeline.
    """
    
    # Add custom stop words to spaCy's stop words list
    for word in custom_stop_words:
        nlp.vocab[word].is_stop = True

    # Remove digits
    df[text_col] = df[text_col].str.replace(r'\d+', '', regex=True)

    # Step 1: Build an NLP pipe
    nlp_pipe = nlp.pipe(df[text_col], batch_size=batch_size, disable=["ner", "parser"])

    # Step 2: Remove punctuation, stopwords, and then lemmatization
    tokens = []
    for doc in nlp_pipe:
        # Filter out punctuation and stop words
        filtered_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        tokens.append(filtered_tokens)
    
    # Filtering out custom stop words (lemmas)
    filtered_tokens = [[token for token in doc if token not in custom_stop_words] 
                       for doc in tokens]
    
    # Join tokens to create a cleaned text column
    df['processed_reviews'] = [' '.join(token) for token in filtered_tokens]

    return df
  
# Custom Preprocessor Class
class Preprocessor:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting required for this preprocessor
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            # Convert Series to DataFrame
            df = X.to_frame(name="Review")
        elif isinstance(X, np.ndarray):
            # Convert NumPy array to DataFrame
            df = pd.DataFrame(X, columns=["Review"])
        else:
            raise ValueError("Input X must be a pandas Series or NumPy array.")

        # Call the preprocess_text function for text cleaning
        preprocess_text(df, text_col="Review", custom_stop_words=custom_stop_words)
        
        # Return the processed reviews as a pandas Series
        return df["processed_reviews"]
