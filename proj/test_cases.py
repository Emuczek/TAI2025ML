import numpy as np
import os
from main import *
from sklearn.model_selection import StratifiedKFold

def run_experiment(X, y, tokenization, feature_repr, strategy, budget):
    if tokenization == "lemmatization+stopword":
        tokenizer = spacy_lemmatizer
    elif tokenization == "whitespace":
        tokenizer = simple_tokenizer
    else:
        raise ValueError(f"Unknown tokenization: {tokenization}")
    X_vect, vectorizer = extract_features(X, tokenizer, vectorizer_type=feature_repr)
    preds, queried_mask = simulate_online_learning(X_vect, y, strategy=strategy, budget=budget)
    metrics = evaluate(preds, y)
    return metrics


def show_result(result_filename):
    print(np.load(result_filename))

if __name__ == "__main__":
    test_cases = {
    "T1": ("lemmatization+stopword", "tfidf", "margin", 0.10),
    "T2": ("lemmatization+stopword", "tfidf", "margin", 0.20),
    "T3": ("lemmatization+stopword", "tfidf", "margin", 0.30),
    "T4": ("whitespace", "tf", "margin", 0.10),
    "T5": ("whitespace", "tf", "margin", 0.20),
    "T6": ("whitespace", "tf", "margin", 0.30),
    "T7": ("lemmatization+stopword", "tfidf", "random", 0.10),
    "T8": ("lemmatization+stopword", "tfidf", "random", 0.20),
    "T9": ("lemmatization+stopword", "tfidf", "random", 0.30),
    "T10": ("lemmatization+stopword", "tfidf", "oracle", 1.00),
}

    os.makedirs("test_results", exist_ok=True)

    csv_path = "data/enron_dataset.csv"
    df = load_enron_csv_dataset(csv_path, max_rows=1000)
    X = df['text']
    y = df['label']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []

    for tid, (tokenization, feature_repr, strategy, budget) in test_cases.items():
        fold_no = 0
        os.makedirs(f"test_results/{tid}", exist_ok=True)
        for train_index, test_index in skf.split(X, y):
            print(f"processing {tid}, fold {fold_no}")
            X_train = X.iloc[train_index].reset_index(drop=True)
            y_train = y.iloc[train_index].reset_index(drop=True)
            X_test = X.iloc[test_index].reset_index(drop=True)
            y_test = y.iloc[test_index].reset_index(drop=True)
            fold_no += 1

            metrics = run_experiment(
                X=X_train,
                y=y_train,
                tokenization=tokenization,
                feature_repr=feature_repr,
                strategy=strategy,
                budget=budget
            )

            fold_metrics.append(metrics)
            save_path = os.path.join("test_results", f"{tid}", f"{tid}_fold_{fold_no}.npy")
            np.save(save_path, metrics)

accuracies = [m['accuracy'] for m in fold_metrics]
print(f"Cross-validated accuracy: {np.mean(accuracies):.4f}")
    




  