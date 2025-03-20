# TripAdvisor-Natural Language Processing-insights for Mini Project 3 (I.O.D 2024)
---
## Problem Statement

The data science team at TripAdvisor is interested to mine insights from hotel reviews on their platform. The reviews are diverse but most reviewers are optimistic about their hotel stay. There are a total of 20,491 reviews and about 73% rated their stay favorably, giving a rating score of 4 or 5.

## Methodology

The reviews are ingested into a Pandas dataframe, and the review text column is analyzed and features are extracted. The rating score provides the ground truth label about the degree of satisfaction about the stay at the hotel. The review text is removed for stop-words, and individual word (token) is lemmatized. The processed reviews are vectorized using Term-Frequency Inverse-Document-Frequency (TF-IDF), and the least and most frequent words in the corpus (Below Top 10 and Top 90 percentile of words (unigrams)) are dropped. Principal Component Analysis is a technique that is used to reduce the dimensionality of the TF-IDF matrix. This allows us to perform visualization of the textual data on the first 2 principal components that explain the most variance of the dataset. Unsupervised techniques such as $k$-means, DBSCAN clustering and Latent Dirichlet Allocation Topic Modeling are performed. Given the review ratings that we used as labels for model training, we used the random forest classifier (RFC) to predict 2 classes (positive and negative sentiments) and 3 classes (positive, neutral and negative sentiments).

## Application Deployment

We used the lightweight option for deployment, so that our internal stakeholders can test the application.

| Platform                                        | URL                                                    |
|-------------------------------------------------|--------------------------------------------------------|
| HuggingFace for serving our RFC model           | https://huggingface.co/michael-map/tripadvisor-nlp-rfc |
| Streamlit Application                           | https://tripadvisor-nlp-insights.streamlit.app/        |
