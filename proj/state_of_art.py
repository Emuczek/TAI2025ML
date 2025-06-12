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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

def run_state_of_art(X, y, tokenization, feature_repr):
    pass

if __name__ == "__main__":

    csv_path = "data/enron_dataset.csv"
    df = load_enron_csv_dataset(csv_path, max_rows=1000)
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    clf = MultinomialNB()
    tokenizer = simple_tokenizer

    fold_metrics = []
    fold_no = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, test_index in skf.split(X, y):
        os.makedirs(f"test_results_state_of_art/fold_{fold_no}", exist_ok=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = MultinomialNB()

        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        print("Cross-validation scores:", cv_scores)
        print("Mean CV accuracy:", np.mean(cv_scores))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = evaluate(y_pred, y_test)
        fold_metrics.append(metrics)
        save_path = os.path.join("test_results_state_of_art", f"fold_{fold_no}", f"fold_{fold_no}.npy")
        np.save(save_path, metrics)

    accuracies = [m['accuracy'] for m in fold_metrics]
    print(f"Cross-validated accuracy: {np.mean(accuracies):.4f}")
