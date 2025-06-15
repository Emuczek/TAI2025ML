from scipy.stats import chisquare
from main import load_enron_csv_dataset
import numpy as np
import os
from scipy.stats import mannwhitneyu

def check_class_balance_chi_square(filepath):
    df = load_enron_csv_dataset(filepath)
    class_counts = df['label'].value_counts()
    count_ham = int(class_counts.get(0, 0))
    count_spam = int(class_counts.get(1, 0))
    expected_occurence = int((count_ham + count_spam) / 2)
    expected = [expected_occurence, expected_occurence]
    observed = [count_ham-1, count_spam]
    print(expected)
    print(observed)
    return chisquare(observed, f_exp=expected)

def show_results(result_filename):
    print(np.load(result_filename))


if __name__ == "__main__":
    # chisquare_res = check_class_balance_chi_square(rf"TAI2025ML\proj\data\enron_dataset.csv")
    # result p-value was 0.5, we accept the null hipothesis that classes are even counted
    # print(chisquare_res)

    ol_acc = []
    ol_f1 = []

    soa_acc = []
    soa_f1 = []

    # online learning
    for test_case in os.listdir("test_results"):
        cnt = 1
        pth = os.path.join("test_results", test_case)
        acc = []
        f1 = []
        recall = []
        precision = []
        for file in os.listdir(pth):
            file_pth = os.path.join(pth, file)
            test_case_results = np.load(file_pth, allow_pickle=True).item()
            acc.append(test_case_results['accuracy'])
            f1.append(test_case_results['f1'])
            recall.append(test_case_results['recall'])
            precision.append(test_case_results['precision'])

        ol_acc.append(round(sum(acc)/len(acc), 2))
        ol_f1.append(round(sum(f1)/len(f1), 2))

        print(f"Test case {test_case}: \n Acccuracy: {round(sum(acc)/len(acc), 2)} +- {round(np.std(acc), 2)}, precission {round(sum(precision)/len(precision), 2)} +- {round(np.std(precision), 2)}, recall {round(sum(recall)/len(recall), 2)} +- {round(np.std(recall), 2)}, f1 score {round(sum(f1)/len(f1), 2)} +- {round(np.std(f1), 2)}")

    # state of art
    for test_case in os.listdir("test_results_state_of_art"):
        pth = os.path.join("test_results_state_of_art", test_case)
        acc = []
        f1 = []
        recall = []
        precision = []
        test_case_results = np.load(pth, allow_pickle=True).item()
        acc.append(test_case_results['accuracy'])
        f1.append(test_case_results['f1'])
        recall.append(test_case_results['recall'])
        precision.append(test_case_results['precision'])

        soa_acc.append(test_case_results['accuracy'])
        soa_f1.append(test_case_results['f1'])

        print(f"Test case {test_case}: \n Acccuracy: {round(sum(acc)/len(acc), 2)} +- {round(np.std(acc), 2)}, precission {round(sum(precision)/len(precision), 2)} +- {round(np.std(precision), 2)}, recall {round(sum(recall)/len(recall), 2)} +- {round(np.std(recall), 2)}, f1 score {round(sum(f1)/len(f1), 2)} +- {round(np.std(f1), 2)}")

    # test for accuracy 
    stat, p_value = mannwhitneyu(ol_acc, soa_acc, alternative='two-sided')

    print(f"UMann-Whitney statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpret result
    alpha = 0.05
    if p_value < alpha:
        print("Statistically significant difference between models.")
    else:
        print("No statistically significant difference between models.")

    # test for f1 score
    stat, p_value = mannwhitneyu(ol_f1, soa_f1, alternative='two-sided')

    print(f"UMann-Whitney statistic: {stat}")
    print(f"P-value: {p_value}")

    # Interpret result
    alpha = 0.05
    if p_value < alpha:
        print("Statistically significant difference between models.")
    else:
        print("No statistically significant difference between models.")