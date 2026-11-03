# Amazon Review Sentiment Analysis

A Machine Learning and Natural Language Processing (NLP) project that analyzes Amazon product reviews and classifies customer opinions into **Positive, Negative, and Neutral** sentiments.

The project uses TF-IDF for text feature extraction and Logistic Regression for sentiment classification. A Streamlit web application provides an interactive interface for entering reviews and viewing predictions.

## Project Objectives

- Analyze customer reviews using NLP techniques.
- Clean and preprocess textual review data.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Evaluate model performance.
- Build an interactive web application using Streamlit.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib
- Jupyter Notebook

## Dataset

Dataset: [Amazon Reviews Sentiment Analysis – Hugging Face](https://huggingface.co/datasets/hugginglearners/amazon-reviews-sentiment-analysis)

- Total original records: 4,915
- Final cleaned records: 4,912
- Input feature: Review Text
- Target variable: Sentiment

### Sentiment Distribution

| Sentiment | Records | Percentage |
|---|---:|---:|
| Positive | 4,449 | 90.52% |
| Negative | 324 | 6.59% |
| Neutral | 142 | 2.89% |

## Project Workflow

1. Dataset loading
2. Exploratory Data Analysis (EDA)
3. Missing value and duplicate removal
4. Sentiment label creation using star ratings
5. Text preprocessing:
   - Lowercase conversion
   - HTML entity handling
   - Punctuation removal
   - Tokenization
   - Stopword removal
6. TF-IDF feature extraction
7. Train-test split (80:20)
8. Logistic Regression model training
9. Model evaluation
10. Streamlit application development

## Machine Learning Model

**Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF Vectorizer

**Maximum TF-IDF Features:** 5,000

**Train-Test Ratio:** 80:20

**Class Handling:** Balanced class weights

## Model Evaluation

Initial Logistic Regression baseline results:

| Metric | Result |
|---|---:|
| Accuracy | 89.73% |
| Negative F1-score | 0.62 |
| Neutral F1-score | 0.06 |
| Positive F1-score | 0.95 |
| Weighted F1-score | 0.90 |
| Macro F1-score | 0.54 |

**Observation:** The model performs well for Positive reviews but has difficulty identifying Neutral reviews. This is partly due to the imbalanced dataset.

These are preliminary baseline results. Further improvements and evaluation are planned.

## Project Structure

```text
amazon-review-sentiment-analysis/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── 01_amazon_sentiment_analysis.ipynb
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Krishna1187/amazon-review-sentiment-analysis.git
```

### 2. Navigate to the project folder

```bash
cd amazon-review-sentiment-analysis
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Future Enhancements

- Improve text preprocessing and model performance.
- Address class imbalance.
- Experiment with Decision Tree, Random Forest, SVM and Naive Bayes.
- Compare multiple machine learning algorithms.
- Experiment with advanced NLP techniques.
- Deploy the application online.
- Improve the user interface and prediction insights.

## Author

**Dr. Shri Krishna Mishra**

Assistant Professor  
Department of Mechanical Engineering

GitHub: [Krishna1187](https://github.com/Krishna1187)

---

*This project is developed as a practical implementation of Machine Learning and Natural Language Processing techniques for customer review sentiment analysis.*str

## Application Preview

### Positive Sentiment Prediction

![Positive Prediction](screenshots/positive_prediction.png)