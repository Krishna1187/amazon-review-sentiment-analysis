
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path

# 1. Create FastAPI application
app = FastAPI(
    title="Amazon Review Sentiment API",
    description="Predict sentiment of product reviews",
    version="1.0"
)

# 2. Load saved ML files
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "models" / "sentiment_model.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

# 3. Define input data structure
class ReviewInput(BaseModel):
    review: str = Field(min_length=1)

# 4. Home API
@app.get("/")
def home():
    return {
        "message": "Amazon Sentiment API is running"
    }

# 5. Prediction API
@app.post("/prediction")
def predict_sentiment(data: ReviewInput):

    # Get review text
    review_text = data.review

    # Convert text into numerical features
    review_vector = vectorizer.transform(
        [review_text]
    )

    # Predict sentiment
    prediction = model.predict(review_vector)[0]

    # Calculate prediction confidence
    probabilities = model.predict_proba(review_vector)[0]

    confidence = float(max(probabilities))

    # Return result
    return {
        "review": review_text,
        "prediction": str(prediction),
        "confidence": round(confidence * 100, 2)
    }
