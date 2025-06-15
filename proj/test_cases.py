import numpy as np
import os
from main import *
from sklearn.model_selection import StratifiedKFold


def run_experiment(X_vect, y, strategy, budget):
    preds, queried_mask = simulate_online_learning(X_vect, y, strategy=strategy, budget=budget)
    metrics = evaluate(preds, y)
    metrics['predictions'] = preds
    metrics['true_labels'] = y
    return metrics


if __name__ == "__main__":
    test_cases = {
        # "T1": ("TK1", "margin", 0.10),
        # "T2": ("TK1", "margin", 0.20),
        # "T3": ("lemmatization+stopword", "tfidf", "margin", 0.30),
        # "T4": ("TK2", "margin", 0.10),
        # "T5": ("TK2", "margin", 0.20),
        # "T6": ("TK2", "margin", 0.30),
        # "T7": ("TK1", "random", 0.10),
        "T8": ("TK1", "random", 0.20),
        "T9": ("TK1", "random", 0.30),
        "T10": ("TK1", "oracle", 1.00),
    }

    token_test_cases = {
        "TK1": ("lemmatization+stopword", "tfidf"),
        "TK2": ("whitespace", "tf"),
    }

    os.makedirs("test_results", exist_ok=True)

    csv_path = "data/enron_dataset.csv"
    df = load_enron_csv_dataset(csv_path)
    X = df['text']
    y = df['label']

    X_vectorized_dict = {}

    for tkid, (tokenization, feature_repr) in token_test_cases.items():
        if tokenization == "lemmatization+stopword":
            tokenizer = spacy_lemmatizer
        elif tokenization == "whitespace":
            tokenizer = simple_tokenizer
        else:
            raise ValueError(f"Unknown tokenization: {tokenization}")
        X_vect_temp, _ = extract_features(X, tokenizer, vectorizer_type=feature_repr)
        X_vectorized_dict[tkid] = X_vect_temp

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    for tid, (token_case, strategy, budget) in test_cases.items():
        fold_no = 0
        os.makedirs(f"test_results/{tid}", exist_ok=True)
        X = X_vectorized_dict[token_case]
        for train_index, test_index in skf.split(X, y):
            print(f"processing {tid}, fold {fold_no}")
            X_train = X.iloc[train_index].reset_index(drop=True)
            y_train = y.iloc[train_index].reset_index(drop=True)
            X_test = X.iloc[test_index].reset_index(drop=True)
            y_test = y.iloc[test_index].reset_index(drop=True)
            fold_no += 1

            metrics = run_experiment(
                X_vect=X_train,
                y=y_train,
                strategy=strategy,
                budget=budget
            )

            fold_metrics.append(metrics)
            save_path = os.path.join("test_results", f"{tid}", f"{tid}_fold_{fold_no}.npy")
            np.save(save_path, metrics)

    accuracies = [m['accuracy'] for m in fold_metrics]
    print(f"Cross-validated accuracy: {np.mean(accuracies):.4f}")
