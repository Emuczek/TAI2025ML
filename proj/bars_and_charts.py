import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd


def plot_metrics(results_df, metric_name, ax):
    """Generates a bar plot for the specified metric."""
    sns.barplot(x="tid", y=metric_name, data=results_df, ax=ax)
    ax.set_title(f"Mean {metric_name} for Each Experiment Variant")
    ax.set_xlabel("Experiment Variant (tid)")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis='x', rotation=45)


def plot_boxplots(results_dict, metric_name, ax):
    """Generates a boxplot for the specified metric."""
    data = [results_dict[tid][metric_name] for tid in results_dict]
    labels = list(results_dict.keys())
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(f"Distribution of {metric_name} for Each Experiment Variant")
    ax.set_xlabel("Experiment Variant (tid)")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis='x', rotation=45)


def plot_roc_auc(results_dict, ax):

    for tid, result in results_dict.items():
        if not result.get('true_labels') or not result.get('predictions') or \
                len(result['true_labels']) == 0 or \
                len(result['true_labels']) != len(result['predictions']):
            print(f"  Skipping {tid}: Brak danych foldów lub niespójne dane.")
            continue
        fold_aucs = []
        for i in range(len(result['true_labels'])):
            try:
                y_true_fold = np.array(result['true_labels'][i]).flatten()
                y_pred_fold = np.array(result['predictions'][i]).flatten()
                if len(np.unique(y_true_fold)) < 2:
                    fold_aucs.append(-1.0)
                    continue

                fpr_fold, tpr_fold, _ = roc_curve(y_true_fold, y_pred_fold)
                fold_aucs.append(auc(fpr_fold, tpr_fold))
            except Exception as e:
                fold_aucs.append(-1.0)
        if not fold_aucs or max(fold_aucs) == -1.0:
            print(
                f"  Nie można było znaleźć najlepszego foldu dla {tid}. Używam foldu 0 (jeśli dostępny) lub pomijam.")
            if len(result['true_labels']) > 0:
                best_fold_idx = 0
                try:
                    if len(np.unique(np.array(result['true_labels'][0]).flatten())) < 2:
                        print(f"  Domyślny fold 0 dla {tid} ma tylko jedną klasę. Pomijam {tid}.")
                        continue
                except Exception:
                    print(f"  Problem z domyślnym foldem 0 dla {tid}. Pomijam {tid}.")
                    continue
            else:
                continue
        else:
            best_fold_idx = np.argmax(fold_aucs)

        fpr, tpr, _ = roc_curve(np.array(result['true_labels'][best_fold_idx]).flatten(), np.array(result['predictions'][best_fold_idx]).flatten())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{tid} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")


def plot_confusion_matrix(results_dict, ax, tid):
    result = results_dict[tid]
    if not result.get('true_labels') or not result.get('predictions') or \
            len(result['true_labels']) == 0 or \
            len(result['true_labels']) != len(result['predictions']):
        print(f"  Skipping {tid}: Brak danych foldów lub niespójne dane.")
    fold_aucs = []
    for i in range(len(result['true_labels'])):
        try:
            y_true_fold = np.array(result['true_labels'][i]).flatten()
            y_pred_fold = np.array(result['predictions'][i]).flatten()
            if len(np.unique(y_true_fold)) < 2:
                fold_aucs.append(-1.0)
            fpr_fold, tpr_fold, _ = roc_curve(y_true_fold, y_pred_fold)
            fold_aucs.append(auc(fpr_fold, tpr_fold))
        except Exception as e:
            fold_aucs.append(-1.0)
    if not fold_aucs or max(fold_aucs) == -1.0:
        print(
            f"  Nie można było znaleźć najlepszego foldu dla {tid}. Używam foldu 0 (jeśli dostępny) lub pomijam.")
        if len(result['true_labels']) > 0:
            best_fold_idx = 0
            try:
                if len(np.unique(np.array(result['true_labels'][0]).flatten())) < 2:
                    print(f"  Domyślny fold 0 dla {tid} ma tylko jedną klasę. Pomijam {tid}.")
            except Exception:
                print(f"  Problem z domyślnym foldem 0 dla {tid}. Pomijam {tid}.")
    else:
        best_fold_idx = np.argmax(fold_aucs)

    """Generates a confusion matrix for the specified experiment variant."""
    result = results_dict[tid]
    cm = confusion_matrix(np.array(result['true_labels'][best_fold_idx]).flatten(), np.array(result['predictions'][best_fold_idx]).flatten())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Confusion Matrix for {tid}')


def main():
    """Main function to load results, generate plots, and save figures."""
    results_dir = "test_results"
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

    results_dict = {}

    for tid in test_cases.keys():
        results_dict[tid] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'predictions': [],
            'true_labels': []
        }
        for fold_no in range(1, 6):
            file_path = os.path.join(results_dir, tid, f"{tid}_fold_{fold_no}.npy")
            try:
                data = np.load(file_path, allow_pickle=True).item()  # Load as a dictionary
                results_dict[tid]['accuracy'].append(data['accuracy'])
                results_dict[tid]['precision'].append(data['precision'])
                results_dict[tid]['recall'].append(data['recall'])
                results_dict[tid]['f1'].append(data['f1'])
                results_dict[tid]['predictions'].append(data['predictions'])
                results_dict[tid]['true_labels'].append(data['true_labels'])
            except FileNotFoundError:
                print(f"File {file_path} not found.")
                continue

    # Create a DataFrame for easier plotting
    results_data = []
    for tid, metrics in results_dict.items():
        for i in range(len(metrics['accuracy'])):
            results_data.append({
                'tid': tid,
                'accuracy': metrics['accuracy'][i],
                'precision': metrics['precision'][i],
                'recall': metrics['recall'][i],
                'f1': metrics['f1'][i]
            })
    results_df = pd.DataFrame(results_data)

    # Bar plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_metrics(results_df, 'accuracy', axes[0, 0])
    plot_metrics(results_df, 'precision', axes[0, 1])
    plot_metrics(results_df, 'recall', axes[1, 0])
    plot_metrics(results_df, 'f1', axes[1, 1])
    plt.tight_layout()
    plt.savefig("metric_barplots.png")
    plt.show()

    # Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_boxplots(results_dict, 'accuracy', axes[0, 0])
    plot_boxplots(results_dict, 'precision', axes[0, 1])
    plot_boxplots(results_dict, 'recall', axes[1, 0])
    plot_boxplots(results_dict, 'f1', axes[1, 1])
    plt.tight_layout()
    plt.savefig("metric_boxplots.png")
    plt.show()

    # ROC curves and AUC
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc_auc(results_dict, ax)
    plt.savefig("roc_auc.png")
    plt.show()

    # Confusion matrices
    selected_tests = ['T3', 'T5', 'T9', 'T10']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, tid in enumerate(selected_tests):
        ax = axes[i // 2, i % 2]
        print(ax)
        print(tid)
        plot_confusion_matrix(results_dict, ax, tid)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.show()


if __name__ == "__main__":

    main()
