import streamlit as st
import pandas as pd

from huggingface_hub import hf_hub_download
import joblib
import spacy

REPO_ID = "michael-map/tripadvisor-nlp-rfc"
FILENAME = "random_forest_model.joblib"

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
    
def predict_sentiment(new_review):
    # Combine SpaCy stopwords with custom stopwords
    combined_stop_words = set(custom_stop_words) | spacy_stop_words
    
    new_review = pd.Series([new_review.lower()])  # Wrap the string in a list to create a Series
    new_review = new_review.apply(preprocess_text)
    
    # Assign the cleaned reviews to a new column
    new_review = pd.DataFrame(new_review, columns=['Reviews'])
    
    # Example of applying advanced text preprocessing with custom stopwords
    new_review = preprocess_text(new_review, text_col="Review", custom_stop_words=combined_stop_words)

    # tfidf_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    # tfidf = joblib.load(tfidf_path)
    
    # joblib.dump(tfidf, "tfidf_vectorizer.joblib")
    
    # new_review_tfidf = tfidf.transform(new_review['processed_reviews'])
    
    prediction_dict = {0: "Negative", 1: "Positive"}
    
    # return prediction_dict.get(rf_clf.predict(new_review_tfidf)[0])
    return None

def run():
    """Loads the model from Hugging Face Hub."""
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = joblib.load(model_path)

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text. :)")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    predicted_sentiment = ""
    if st.button("Predict"):
        predicted_sentiment = model.predict(pd.Series(userinput))
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        sentiment=f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()
