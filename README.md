# 🎬 IMDB Sentiment Analysis with Random Forest

This project performs **sentiment classification** on the IMDB movie reviews dataset using Natural Language Processing (NLP) techniques and a **Random Forest Classifier**.

> 📌 Predict whether a movie review is **positive** or **negative** based on its text content.

---

## 📚 Dataset

- **Source:** [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews
- **Columns:**
  - `review`: The text of the movie review
  - `sentiment`: Sentiment label — `'positive'` or `'negative'`

---

## 🧰 Libraries Used

- `pandas`, `numpy` – data manipulation  
- `re`, `html` – text cleaning  
- `nltk` – stopwords, stemming, tokenization  
- `sklearn` – TF-IDF, train-test split, modeling, evaluation  

---

## 🧹 Preprocessing Pipeline

### 1. Label Encoding
Convert sentiment strings to binary:
- `'positive'` → `1`
- `'negative'` → `0`

### 2. Text Cleaning
Each review is cleaned using:
- HTML unescaping  
- Removing URLs, mentions, hashtags  
- Removing special characters (non-alphabetic)  
- Lowercasing  
- Removing English stopwords  
- Tokenization with `TweetTokenizer`

### 3. Stemming
Applied **PorterStemmer** to reduce words to their base/root form.

---

## 🔢 Feature Extraction

Used **TF-IDF Vectorization** to convert text into numerical features:
- `max_features=10000`: Top 10,000 terms  
- `ngram_range=(1, 2)`: Unigrams and bigrams  
- `min_df=5`: Ignore rare terms  

---

## 🧪 Train-Test Split

Split the data into:
- `80%` Training  
- `20%` Testing  

---

## 🌲 Model: Random Forest Classifier

Trained a **RandomForestClassifier** from `sklearn.ensemble`.

### ✅ Evaluation Metrics:
- **Precision**: % of predicted positives that are actual positives  
- **Recall**: % of actual positives that are correctly predicted  
- **F1-score**: Harmonic mean of precision and recall  
- **Accuracy**

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)  
- Experiment with other models (Logistic Regression, SVM, LSTM)  
- Use word embeddings (Word2Vec, GloVe)  
