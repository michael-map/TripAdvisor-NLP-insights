import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    prediction_prob = model.predict_proba(pd.Series(review_text))[0]
    
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
            prob_positive = round(prediction_prob[1] * 100, 2)
            prob_negative = round(prediction_prob[0] * 100, 2)
    
            # Display Results
            st.markdown(f"### Sentiment: **{sentiment}**")
            st.markdown(f"**Confidence:** {prob_positive}% Positive, {prob_negative}% Negative")
            
            # Plotly Bar Chart for Probabilities
            fig = go.Figure(data=[
                go.Bar(
                    x=["Positive", "Negative"],
                    y=[prob_positive, prob_negative],
                    text=[f"{prob_positive}%", f"{prob_negative}%"],
                    textposition='auto',
                    marker=dict(color=['green', 'red'])
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Sentiment",
                yaxis_title="Probability (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig)
            
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

