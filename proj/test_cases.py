import numpy as np
import os
from main import *

# Simulated experiment function (replace with your actual logic)
def run_experiment(tokenization, feature_repr, strategy, budget):

    csv_path = "data/enron_dataset.csv"
    df = load_enron_csv_dataset(csv_path, max_rows=1000)

    if tokenization == "lemmatization+stopword":
        tokenizer = spacy_lemmatizer
    elif tokenization == "whitespace":
        tokenizer = simple_tokenizer

    X_vect, vectorizer = extract_features(df['text'], tokenizer, vectorizer_type=feature_repr)
    preds, queried_mask = simulate_online_learning(X_vect, df['label'].to_numpy(), budget=budget)

    metrics = evaluate(preds, df['label'].to_numpy())

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

    # for tid, (tokenization, feature_repr, strategy, budget) in test_cases.items():
    #     print(f"Running {tid}...")
    #     result = run_experiment(tokenization, feature_repr, strategy, budget)
    #     save_path = os.path.join("test_results", f"{tid}.npy")
    #     np.save(save_path, result)

    for test_result_file in os.listdir("test_results"):
        print(np.load(test_result_file))
