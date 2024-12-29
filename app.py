import streamlit as st
import pandas as pd

from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "michael-map/tripadvisor-nlp-rfc"
FILENAME = "random_forest_model.joblib"

def run():
    """Loads the model from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

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
