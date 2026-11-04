
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="⭐",
    layout="centered"
)

# Application title
st.title("⭐ Amazon Review Sentiment Analysis")

st.write(
    "Enter your Amazon product review "
    "to predict its sentiment."
)

# Text input
review = st.text_area(
    "Enter Product Review",
    placeholder="Write your product review here..."
)

# Prediction button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        try:
            # Send request to FastAPI
            response = requests.post(
                "http://127.0.0.1:8000/prediction",
                json={"review": review},
                timeout=30
            )

            # Check API response
            response.raise_for_status()

            # Convert JSON response into Python dictionary
            result = response.json()

            # Display prediction
            st.subheader("Prediction Result")

            sentiment = result["prediction"]

            if sentiment == "Positive":
                st.success("Positive Review 😊")

            elif sentiment == "Negative":
                st.error("Negative Review 😞")

            else:
                st.info("Neutral Review 😐")

            st.write(
                "Confidence:",
                str(result["confidence"]) + "%"
            )

        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: {e}")

        except (KeyError, ValueError):
            st.error("Invalid response received from API.")
