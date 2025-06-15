'''

State of the art method - spam classification with Multinomial Naive Bayes classifier

'''
from sklearn.naive_bayes import MultinomialNB
from main import *
import os 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score

def run_state_of_art(tokenization, feature_repr, tid):
    csv_path = "data/enron_dataset.csv"

    df = load_enron_csv_dataset(csv_path)

    tokenizer = None
    if tokenization == "lemmatization+stopword":
        tokenizer = spacy_lemmatizer
    elif tokenization == "whitespace":
        tokenizer = simple_tokenizer
    else:
        raise ValueError(f"Unknown tokenization: {tokenization}")
    
    X_vect, vectorizer= extract_features(df['text'], tokenizer, vectorizer_type=feature_repr)
    y = df['label']

    clf = MultinomialNB()
    fold_metrics = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, test_index in skf.split(X_vect, y):
        X_train, X_test = X_vect[train_index], X_vect[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = evaluate(y_pred, y_test)
        fold_metrics.append(metrics)
        save_path = os.path.join("test_results_state_of_art", f"{tid}.npy")
        np.save(save_path, metrics)

    # accuracies = [m['accuracy'] for m in fold_metrics]
    # print(f"Cross-validated accuracy: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    test_cases = {
    "T1": ("lemmatization+stopword", "tfidf"),
    "T2": ("lemmatization+stopword", "tf"),
    "T3": ("whitespace", "tf"),
    "T4": ("whitespace", "tf"),
    }
    for tid, (tokenization, feature_repr) in test_cases.items():
        print(f"Processing {tid}")
        run_state_of_art(tokenization, feature_repr, tid)



