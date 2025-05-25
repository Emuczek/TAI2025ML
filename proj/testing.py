from scipy.stats import chisquare
from main import load_enron_csv_dataset

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


if __name__ == "__main__":
    chisquare_res = check_class_balance_chi_square(rf"TAI2025ML\proj\data\enron_dataset.csv")
    # result p-value was 0.5, we accept the null hipothesis that classes are even counted
    print(chisquare_res)