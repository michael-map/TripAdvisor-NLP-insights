import streamlit as st
import pandas as pd

from huggingface_hub import hf_hub_download
import joblib

from utils import Preprocessor

REPO_ID = "michael-map/tripadvisor-nlp-rfc"
FILENAME = "random_forest_model.joblib"

# def run():
#     """Loads the model from Hugging Face Hub."""
    
#     model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
#     model = joblib.load(model_path)

#     st.title("Sentiment Analysis")
#     st.text("Basic app to detect the sentiment of text. :)")
#     st.text("")
#     userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
#     st.text("")
#     predicted_sentiment = ""
#     if st.button("Predict"):
#         predicted_sentiment = model.predict(pd.Series(userinput))
#         if predicted_sentiment == 1:
#             output = 'positive 👍'
#         else:
#             output = 'negative 👎'
#         sentiment=f'Predicted sentiment of "{userinput}" is {output}.'
#         st.success(sentiment)


import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Helper function for prediction
def predict_review(review_text):
    
    # Predict sentiment
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = joblib.load(model_path)
    prediction = model.predict(pd.Series(review_text))
    prediction_prob = model.predict_proba(pd.Series(review_text))
    
    return prediction, prediction_prob

def run():
    # Streamlit UI
    st.set_page_config(page_title="Hotel Review Sentiment Predictor", layout="centered")
    
    # Header
    st.title("Hotel Review Sentiment Predictor")
    st.subheader("Analyze and predict the sentiment of hotel reviews.")
    
    # User Input
    st.markdown("### Enter Your Review")
    user_review = st.text_area(
        "Type or paste a hotel review below to predict its sentiment.",
        placeholder="The room was clean, and the service was excellent!",
    )
    
    # Submit Button
    if st.button("Predict Sentiment"):
        if user_review.strip():
            # Make prediction
            prediction, prediction_prob = predict_review(user_review)
            sentiment = "Positive" if prediction == 1 else "Negative"
            prob_positive = prediction_prob #round(prediction_prob * 100, 2)
            prob_negative = prediction_prob #round(prediction_prob * 100, 2)
    
            # Display Results
            st.markdown(f"### Sentiment: **{sentiment}**")
            st.markdown(f"**Confidence:** {prob_positive}% Positive, {prob_negative}% Negative")
            st.info(
                "Sentiment prediction is based on trained machine learning algorithms using advanced text processing techniques."
            )
        else:
            st.error("Please enter a valid review before clicking 'Predict Sentiment'.")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit | © 2024 Hotel Insights AI")

if __name__ == "__main__":
    run()

