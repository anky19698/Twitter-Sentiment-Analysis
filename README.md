# Twitter Sentiment Analysis

This project aims to perform sentiment analysis on tweets using various machine learning models. The goal is to classify the sentiment of tweets into four categories: Positive, Negative, Neutral, and Irrelevant.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)

## Project Overview

This project aims to analyze the sentiment of tweets using natural language processing (NLP) techniques and machine learning algorithms. The main steps include:

1. Data preprocessing and cleaning
2. Feature extraction using TF-IDF vectorization
3. Training and evaluating multiple machine learning models
4. Hyperparameter tuning using GridSearchCV
5. Model selection and final evaluation on a validation set

## Dependencies

- numpy
- pandas
- nltk
- scikit-learn
- spacy
- xgboost

You can install the necessary libraries using the following command:

```bash
pip install numpy pandas nltk scikit-learn spacy xgboost
```

## Data

The project uses two datasets:
- `twitter_training.csv`: Main dataset for training and testing
- `twitter_validation.csv`: Validation dataset for final model evaluation

Both datasets contain columns: tweet_id, entity, sentiment, and tweet_text.

## Preprocessing

Text preprocessing is done using spaCy, including:
- Tokenization
- Lemmatization
- Removal of stop words and punctuation

## Models

The following machine learning models are implemented and evaluated:

1. Multinomial Naive Bayes
2. Random Forest Classifier
3. Logistic Regression
4. XGBoost Classifier
5. Support Vector Machine (SVM)

GridSearchCV is used to find the best hyperparameters for each model.

## Results

The Random Forest Classifier achieved the best performance on the test set:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.97      | 0.84   | 0.90     | 2696    |
| 1     | 0.85      | 0.95   | 0.90     | 4119    |
| 2     | 0.91      | 0.94   | 0.92     | 4380    |
| 3     | 0.94      | 0.89   | 0.91     | 3605    |

| Metric       | Score |
|--------------|-------|
| Accuracy     | 0.91  |
| Macro Avg    | 0.91  |
| Weighted Avg | 0.91  |

On the validation set, the model achieved:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.98      | 0.90   | 0.94     | 172     |
| 1     | 0.95      | 0.95   | 0.95     | 277     |
| 2     | 0.94      | 0.96   | 0.95     | 266     |
| 3     | 0.93      | 0.95   | 0.94     | 285     |

| Metric       | Score |
|--------------|-------|
| Accuracy     | 0.94  |
| Macro Avg    | 0.94  |
| Weighted Avg | 0.94  |

## Usage

1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook to train and evaluate the models
4. Use the trained model (saved as `model.pkl`) for sentiment prediction on new data

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Preprocess and predict
def predict_sentiment(text):
    processed_text = process_text(text)  # Implement the process_text function as in the notebook
    return model.predict([processed_text])[0]

# Example usage
sentiment = predict_sentiment("I love this new product!")
print(sentiment)
```
