# spam_classifier_project.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy
import random

nltk.download("punkt")
nltk.download("stopwords")
spacy_model = spacy.load("en_core_web_sm")

def simple_tokenizer(text):
    return text.lower().split()

def nltk_tokenizer(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
    return tokens

def spacy_lemmatizer(text):
    doc = spacy_model(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords.words('english')]

def load_enron_csv_dataset(csv_path, max_rows=None):
    df = pd.read_csv(csv_path)
    df = df[['Subject', 'Message', 'Spam/Ham']].dropna()
    df['text'] = df['Subject'].astype(str) + ' ' + df['Message'].astype(str)
    df['label'] = df['Spam/Ham'].str.lower().map({'spam': 1, 'ham': 0})
    df = df[['text', 'label']]

    if max_rows:
        df = df.sample(n=max_rows, random_state=42)

    df = df.dropna(subset=['text', 'label'])
    return df.reset_index(drop=True)

def extract_features(texts, tokenizer, vectorizer_type='tfidf'):
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    else:
        vectorizer = CountVectorizer(tokenizer=tokenizer)
    return vectorizer.fit_transform(texts), vectorizer

def simulate_online_learning(X, y, budget=0.2, strategy="margin"):
    clf = PassiveAggressiveClassifier(max_iter=1, tol=1e-3, random_state=42)
    queried = 0
    n = X.shape[0]
    max_queries = int(budget * n)

    predictions = []
    queried_mask = []

    for i in range(n):
        x_i = X[i].reshape(1, -1)
        if queried == 0:
            clf.partial_fit(x_i, [y[i]], classes=[0, 1])
            pred = clf.predict(x_i)
            predictions.append(pred[0])
            queried += 1
            queried_mask.append(True)
            continue

        pred = clf.predict(x_i)
        predictions.append(pred[0])

        query = False

        if queried < max_queries:
            if strategy == "random":
                query = random.random() < (budget)
            elif strategy == "margin":
                if hasattr(clf, "decision_function"):
                    margin = abs(clf.decision_function(x_i)[0])
                    query_probability = 1 / (1 + margin)
                    query = random.random() < query_probability
                else:
                    query = False
            elif strategy == "oracle":
                query = True 
        if query:
            clf.partial_fit(x_i, [y[i]])
            queried += 1
            queried_mask.append(True)
        else:
            queried_mask.append(False)

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"Processed {i+1}/{n} samples...", file=sys.stderr)

    return predictions, queried_mask


def evaluate(preds, y_true):
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds)
    }

def main():
    csv_path = "data/enron_dataset.csv"
    df = load_enron_csv_dataset(csv_path, max_rows=1000)

    tokenizer = spacy_lemmatizer

    X_vect, vectorizer = extract_features(df['text'], tokenizer)
    preds, queried_mask = simulate_online_learning(X_vect, df['label'].to_numpy(), budget=0.3)

    results = evaluate(preds, df['label'].to_numpy())
    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print(f"Queried {sum(queried_mask)} of {len(df)} samples")

if __name__ == "__main__":
    main()
