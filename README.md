# OIBSIP — Machine Learning Internship Projects

A collection of machine learning projects completed during the **Oasis Infobyte Data Science Internship**.

The repository contains three independent projects covering regression, text classification, and multiclass classification.

## Projects

### 1. Car Price Prediction

Predicts the estimated selling price of a used car from features such as purchase year, present price, kilometers driven, fuel type, seller type, transmission type, and ownership history.

**Model:** Random Forest Regressor

**Key steps:**

- Data preprocessing
- Feature engineering
- One-hot encoding
- Train/test split
- Regression model training
- R², MAE, and RMSE evaluation
- Model serialization
- Streamlit prediction interface

**Run:**

```bash
python Car-Price-Prediction/src/train_model.py
streamlit run Car-Price-Prediction/src/app.py
```

---

### 2. Email Spam Detection

Classifies email/message text as **Spam** or **Not Spam**.

**Model:** Multinomial Naive Bayes  
**Text representation:** TF-IDF with unigram and bigram features

**Key steps:**

- Text cleaning
- TF-IDF vectorization
- Stratified train/test split
- Naive Bayes classification
- Accuracy and classification report
- Confusion matrix
- ROC curve / AUC
- Saved model + vectorizer
- Flask prediction interface

**Run:**

```bash
cd email_spam_detection
pip install -r requirements.txt
python src/train.py
python app.py
```

---

### 3. Iris Flower Detection

Predicts the species of an Iris flower from sepal and petal measurements.

**Model:** SVM with RBF kernel

**Key steps:**

- Label encoding
- Standard scaling
- SVM pipeline
- Stratified train/test split
- Accuracy evaluation
- Classification report
- Confusion matrix
- Saved model bundle
- Streamlit prediction interface

**Train:**

```bash
python iris_flower_detection/src/train.py
```

**Run UI:**

```bash
streamlit run iris_flower_detection/src/app.py
```

> The current Iris Streamlit app contains a local model-file path. Update that path to the generated `iris_model.pkl` location on your machine before running.

## Repository Structure

```text
OIBSIP/
├── Car-Price-Prediction/
│   ├── data/
│   ├── model/
│   └── src/
├── email_spam_detection/
│   ├── data/
│   ├── model/
│   ├── src/
│   ├── app.py
│   └── requirements.txt
└── iris_flower_detection/
    ├── data/
    ├── model/
    └── src/
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Flask
- Joblib / Pickle

## Learning Outcomes

These projects demonstrate practical use of:

- Regression
- Classification
- NLP preprocessing
- Feature engineering
- Model evaluation
- Model persistence
- Lightweight ML web interfaces

## Author

**Sankalp Gupta**  
Data Science Intern — Oasis Infobyte

GitHub: https://github.com/Sankalp-gupta1
