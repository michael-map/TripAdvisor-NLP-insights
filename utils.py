from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import joblib

custom_stop_words = ["hotel", "room", "day", "night", "time", "got",
                    "people", "asked", "told", "thing", "really", "want",
                    "go", "stay", "come", "look", "say", "try", "think",
                    "service", "nice", "place", "know", "little", "check", 
                    "small", "bit", "big", "lot", "n't", "way", "close", "work", "need"]

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
