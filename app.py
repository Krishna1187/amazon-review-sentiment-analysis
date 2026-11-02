
import streamlit as st
import joblib
import os
import pandas as pd

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)

# --------------------------------
# 2. Load Trained Model
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
)

# --------------------------------
# 3. Application Heading
# --------------------------------

st.title("🛒 Amazon Review Sentiment Analysis")

st.write(
    "Enter a customer review to predict whether "
    "the sentiment is Positive, Negative or Neutral."
)

st.divider()

# --------------------------------
# 4. User Input
# --------------------------------

review = st.text_area(
    "Enter Amazon Product Review:",
    placeholder="Example: Excellent product, very good quality...",
    height=150
)

# --------------------------------
# 5. Sentiment Prediction
# --------------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        # Convert review into TF-IDF features
        review_vector = tfidf.transform([review])

        # Predict sentiment
        prediction = model.predict(review_vector)[0]

        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success("Positive Sentiment 😊")

        elif prediction == "Negative":
            st.error("Negative Sentiment 😞")

        else:
            st.info("Neutral Sentiment 😐")

st.divider()

st.caption("Machine Learning Model: Logistic Regression + TF-IDF")
