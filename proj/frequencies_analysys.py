import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, kstest, \
    power_divergence  # G-test is power_divergence with lambda_='log-likelihood'
from sklearn.feature_extraction.text import CountVectorizer
import os

# --- (Optional) Download NLTK resources if not already present ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt')


# --- 1. Load and Preprocess Data ---
def load_enron_csv_dataset(csv_path, max_rows=None):
    """
    Loads the Enron dataset from a CSV file.
    Expected CSV structure: 'Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date'
    """
    try:
        # Specify encoding if default (utf-8) fails. Common ones: 'latin1', 'iso-8859-1', 'cp1252'
        df = pd.read_csv(csv_path, nrows=max_rows, encoding='latin1')  # Using latin1 as it's common for Enron
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return pd.DataFrame()  # Return empty DataFrame
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return pd.DataFrame()

    # Define expected column names based on user's CSV structure
    text_col = 'Message'
    label_col = 'Spam/Ham'

    # Check for expected columns
    if text_col not in df.columns:
        print(f"Error: Expected text column '{text_col}' not found in CSV. Found columns: {df.columns.tolist()}")
        return pd.DataFrame()
    if label_col not in df.columns:
        print(f"Error: Expected label column '{label_col}' not found in CSV. Found columns: {df.columns.tolist()}")
        return pd.DataFrame()

    df_processed = pd.DataFrame()
    df_processed['text'] = df[text_col].astype(str)  # Ensure text is string

    # Convert labels to numeric (0 for ham, 1 for spam)
    # Assuming 'Spam/Ham' column contains 'spam' or 'ham' strings (case-insensitive check)
    original_labels_series = df[label_col].astype(str).str.lower()
    df_processed['label'] = original_labels_series.apply(lambda x: 1 if x == 'spam' else 0)

    # Verification and warnings
    unique_original_labels = original_labels_series.unique()
    expected_labels = ['spam', 'ham']

    # Check if ONLY expected labels are present
    if not all(label in expected_labels for label in unique_original_labels):
        unexpected_found = [label for label in unique_original_labels if label not in expected_labels]
        print(f"Warning: Unexpected values found in label column '{label_col}'. Values: {unexpected_found}. "
              f"Only 'spam' is mapped to 1 (Spam); all other values (including 'ham' and unexpected) are mapped to 0 (Ham).")

    if df_processed.empty:
        print("Warning: No data processed, possibly due to incorrect column names or empty CSV.")
        return df_processed  # Return empty DataFrame

    spam_count = df_processed['label'].sum()
    ham_count = len(df_processed) - spam_count
    print(f"Loaded {len(df_processed)} rows. Processed Spam: {spam_count}, Processed Ham: {ham_count}")

    if len(df_processed) > 0:  # Only print these warnings if data was loaded
        if spam_count == 0 and ham_count > 0:
            print(
                "Warning: Dataset contains only Ham messages after processing labels. Some analyses might not be meaningful.")
        elif ham_count == 0 and spam_count > 0:
            print(
                "Warning: Dataset contains only Spam messages after processing labels. Some analyses might not be meaningful.")
        elif spam_count == 0 and ham_count == 0:  # This case implies labels were not 'spam' or 'ham', or rows were filtered
            print(
                "Warning: No Spam or Ham messages identified after processing. Check label column content and mapping.")

    return df_processed


# --- Text Preprocessing ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    if not isinstance(text, str):  # Handle potential non-string data
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if
              word not in stop_words and len(word) > 2]  # Stem and remove stop words
    return " ".join(tokens)


# --- Directory for saving plots ---
output_dir = "enron_analysis_plots"
os.makedirs(output_dir, exist_ok=True)


# --- Main Analysis ---
def run_analysis(df_path):
    print(f"Loading dataset from: {df_path}")
    df = load_enron_csv_dataset(df_path)

    if df.empty:
        print("Dataset is empty or could not be loaded. Aborting analysis.")
        return pd.DataFrame(columns=['word', 'spam_freq', 'ham_freq', 'g_statistic',
                                     'p_value'])  # Return empty DF with expected columns

    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Separate spam and ham
    spam_texts = df[df['label'] == 1]['processed_text']
    ham_texts = df[df['label'] == 0]['processed_text']

    if spam_texts.empty and ham_texts.empty:
        print("No text data found for either spam or ham categories after preprocessing. Aborting.")
        return pd.DataFrame(columns=['word', 'spam_freq', 'ham_freq', 'g_statistic', 'p_value'])
    if spam_texts.empty:
        print("Warning: No text data found for spam category. Some analyses will be skipped.")
    if ham_texts.empty:
        print("Warning: No text data found for ham category. Some analyses will be skipped.")

    # --- 2. Calculate Term Frequencies ---
    spam_term_freqs = pd.Series(dtype=int)
    if not spam_texts.empty:
        vectorizer_spam = CountVectorizer(min_df=1)  # min_df=1 ensures it works even with few documents
        try:
            spam_dtm = vectorizer_spam.fit_transform(spam_texts.dropna())  # dropna just in case
            if spam_dtm.shape[1] > 0:  # if vocabulary is not empty
                spam_term_freqs = pd.Series(np.array(spam_dtm.sum(axis=0)).flatten(),
                                            index=vectorizer_spam.get_feature_names_out()).sort_values(ascending=False)
            else:
                print("No features found for spam texts after vectorization.")
        except ValueError as e:
            print(f"Error vectorizing spam texts (possibly all empty after preprocessing): {e}")

    ham_term_freqs = pd.Series(dtype=int)
    if not ham_texts.empty:
        vectorizer_ham = CountVectorizer(min_df=1)
        try:
            ham_dtm = vectorizer_ham.fit_transform(ham_texts.dropna())
            if ham_dtm.shape[1] > 0:
                ham_term_freqs = pd.Series(np.array(ham_dtm.sum(axis=0)).flatten(),
                                           index=vectorizer_ham.get_feature_names_out()).sort_values(ascending=False)
            else:
                print("No features found for ham texts after vectorization.")
        except ValueError as e:
            print(f"Error vectorizing ham texts (possibly all empty after preprocessing): {e}")

    print(f"\nTop 10 Spam words (if any):\n{spam_term_freqs.head(10)}")
    print(f"\nTop 10 Ham words (if any):\n{ham_term_freqs.head(10)}")

    # --- 3. Visualizations ---
    def plot_top_n_words(term_freqs, title, filename, n=20):
        if term_freqs.empty or len(term_freqs) < 1:
            print(f"Skipping plot '{title}' as no term frequencies are available.")
            return
        plt.figure(figsize=(12, 8))
        term_freqs.head(n).plot(kind='bar')
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xlabel("Term")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_top_n_words(spam_term_freqs, "Top 20 Most Frequent Words in Spam Emails", "top_spam_words.png")
    plot_top_n_words(ham_term_freqs, "Top 20 Most Frequent Words in Ham Emails", "top_ham_words.png")

    def generate_wordcloud(term_freqs, title, filename):
        if term_freqs.empty:
            print(f"Cannot generate word cloud for {title}, no terms found.")
            return
        wc_dict = term_freqs.to_dict()
        if not wc_dict:
            print(f"Cannot generate word cloud for {title}, term frequency dictionary is empty.")
            return
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wc_dict)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        except ValueError as e:  # Can happen if all frequencies are zero after conversion
            print(f"Error generating word cloud for {title}: {e}")

    generate_wordcloud(spam_term_freqs, "Word Cloud for Spam Emails", "spam_wordcloud.png")
    generate_wordcloud(ham_term_freqs, "Word Cloud for Ham Emails", "ham_wordcloud.png")

    def plot_zipf(term_freqs, label_name, filename):  # Changed 'label' to 'label_name' to avoid conflict
        if term_freqs.empty or len(term_freqs) < 2:  # Need at least 2 points for a meaningful log-log plot
            print(f"Skipping Zipf plot for {label_name} as there are insufficient term frequencies.")
            return
        ranks = np.arange(1, len(term_freqs) + 1)
        frequencies = term_freqs.values
        plt.figure(figsize=(8, 6))
        plt.loglog(ranks, frequencies, marker=".")
        plt.title(f"Zipf's Law Plot for {label_name} Emails")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_zipf(spam_term_freqs, "Spam", "zipf_spam.png")
    plot_zipf(ham_term_freqs, "Ham", "zipf_ham.png")

    plt.figure(figsize=(10, 7))
    plot_combined_zipf = False
    if not spam_term_freqs.empty and len(spam_term_freqs) >= 2:
        ranks_spam = np.arange(1, len(spam_term_freqs) + 1)
        plt.loglog(ranks_spam, spam_term_freqs.values, marker=".", label="Spam")
        plot_combined_zipf = True
    if not ham_term_freqs.empty and len(ham_term_freqs) >= 2:
        ranks_ham = np.arange(1, len(ham_term_freqs) + 1)
        plt.loglog(ranks_ham, ham_term_freqs.values, marker=".", label="Ham")
        plot_combined_zipf = True

    if plot_combined_zipf:
        plt.title("Combined Zipf's Law Plot")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "zipf_combined.png"))
    else:
        print("Skipping combined Zipf plot due to insufficient data in one or both categories.")
    plt.close()

    def plot_freq_distribution(term_freqs, label_name, filename):  # Changed 'label' to 'label_name'
        if term_freqs.empty:
            print(f"Skipping frequency distribution plot for {label_name} as term frequencies are empty.")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(term_freqs.values,
                 bins=max(1, min(50, len(term_freqs.values) // 2 if len(term_freqs.values) > 3 else 1)), log=True,
                 alpha=0.7)
        plt.title(f"Distribution of Word Frequencies in {label_name} Emails")
        plt.xlabel("Word Frequency")
        plt.ylabel("Number of Words (log scale)")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_freq_distribution(spam_term_freqs, "Spam", "freq_dist_spam.png")
    plot_freq_distribution(ham_term_freqs, "Ham", "freq_dist_ham.png")

    # --- 4. Statistical Tests ---
    print("\n--- Statistical Tests ---")

    combined_freq_df = pd.DataFrame({'spam': spam_term_freqs, 'ham': ham_term_freqs}).fillna(0).astype(int)
    total_spam_words_corpus = spam_term_freqs.sum() if not spam_term_freqs.empty else 0
    total_ham_words_corpus = ham_term_freqs.sum() if not ham_term_freqs.empty else 0

    g_test_results = []
    g_test_df = pd.DataFrame(columns=['word', 'spam_freq', 'ham_freq', 'g_statistic', 'p_value'])  # Default empty

    if total_spam_words_corpus > 0 and total_ham_words_corpus > 0 and not combined_freq_df.empty:
        for word, freqs in combined_freq_df.iterrows():
            count_word_spam = freqs['spam']
            count_word_ham = freqs['ham']

            # Ensure these are non-negative for subtraction
            # total_spam_words_corpus and total_ham_words_corpus are sums of frequencies, always >= individual counts

            observed = np.array([
                [count_word_spam, count_word_ham],
                [total_spam_words_corpus - count_word_spam, total_ham_words_corpus - count_word_ham]
            ])

            # The G-test requires all marginal sums to be non-zero.
            # If a word count equals total words (e.g., only one word in corpus), the second row becomes zero.
            if np.any(observed < 0) or np.any(observed.sum(axis=1) == 0) or np.any(observed.sum(axis=0) == 0):
                # print(f"Skipping G-test for word '{word}' due to zero marginal sums or negative counts. Observed: {observed.tolist()}")
                # We can still store it with NaN for results, as per original logic for ValueError
                g_test_results.append({
                    'word': word,
                    'spam_freq': int(count_word_spam),
                    'ham_freq': int(count_word_ham),
                    'g_statistic': np.nan,
                    'p_value': np.nan
                })
            continue

            try:
                g_stat, p_value, dof, expected = power_divergence(observed, lambda_='log-likelihood')
                g_test_results.append({
                    'word': word,
                    'spam_freq': int(count_word_spam),
                    'ham_freq': int(count_word_ham),
                    'g_statistic': g_stat,
                    'p_value': p_value
                })
            except ValueError as e:
                # print(f"Could not perform G-test for word '{word}': {e}. Observed: {observed.tolist()}")
                g_test_results.append({
                    'word': word,
                    'spam_freq': int(count_word_spam),
                    'ham_freq': int(count_word_ham),
                    'g_statistic': np.nan,
                    'p_value': np.nan
                })

        if g_test_results:
            g_test_df = pd.DataFrame(g_test_results).sort_values(by='p_value').dropna(subset=['g_statistic', 'p_value'])
            print("\nTop words significantly different between Spam and Ham (G-test, sorted by p-value):")
            print(g_test_df.head(20))
            g_test_df.to_csv(os.path.join(output_dir, "g_test_word_significance.csv"), index=False)
        else:
            print("No valid G-test results to report.")

    else:
        print(
            "Skipping G-test as one or both corpora (spam/ham) are empty, have no words, or combined frequency data is missing.")

    if not spam_term_freqs.empty and not ham_term_freqs.empty:
        ks_stat, ks_p_value = kstest(spam_term_freqs.values, ham_term_freqs.values)
        print(f"\nKolmogorov-Smirnov test on frequency distributions:")
        print(f"  KS Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_p_value:.4g}")
        if ks_p_value < 0.05:
            print("  The distributions of word frequencies are significantly different (p < 0.05).")
        else:
            print("  No significant difference found in the distributions of word frequencies (p >= 0.05).")
    else:
        print("\nSkipping Kolmogorov-Smirnov test as one or both term frequency lists are empty.")

    if not spam_term_freqs.empty and not ham_term_freqs.empty:
        try:
            # Mann-Whitney U requires at least one observation in each sample that is not tied with all observations in the other sample.
            # And inputs should not be all identical.
            if len(np.unique(spam_term_freqs.values)) > 1 or len(np.unique(ham_term_freqs.values)) > 1 or \
                    not np.array_equal(spam_term_freqs.values, ham_term_freqs.values):

                mwu_stat, mwu_p_value = mannwhitneyu(spam_term_freqs.values, ham_term_freqs.values,
                                                     alternative='two-sided')
                print(f"\nMann-Whitney U test on frequency values:")
                print(f"  U Statistic: {mwu_stat:.4f}")
                print(f"  P-value: {mwu_p_value:.4g}")
                if mwu_p_value < 0.05:
                    median_freq_spam = np.median(spam_term_freqs.values)
                    median_freq_ham = np.median(ham_term_freqs.values)
                    print(f"  Median frequency in Spam: {median_freq_spam}, Median frequency in Ham: {median_freq_ham}")
                    direction = "higher" if median_freq_spam > median_freq_ham else (
                        "lower" if median_freq_spam < median_freq_ham else "different but with same median")
                    print(
                        f"  The word frequencies in one group tend to be significantly {direction} than in the other (p < 0.05).")
                else:
                    print("  No significant difference found in the central tendency of word frequencies (p >= 0.05).")
            else:
                print("\nSkipping Mann-Whitney U test: Input arrays are identical or lack variability.")

        except ValueError as e:
            print(f"\nMann-Whitney U test could not be performed: {e}")
    else:
        print("\nSkipping Mann-Whitney U test as one or both term frequency lists are empty.")

    print(f"\nAll plots and G-test results (if any) saved to '{output_dir}' directory.")
    return g_test_df


# --- Execute the analysis ---
if __name__ == "__main__":
    # USER: This should be the path to your Enron dataset CSV
    # The CSV structure is expected to be: 'Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date'
    csv_path_to_use = "data/enron_dataset.csv"

    # Ensure the directory for the CSV exists (if nested, e.g., "data/")
    # This does not create the CSV file itself.
    csv_dir = os.path.dirname(csv_path_to_use)
    if csv_dir and not os.path.exists(csv_dir):  # If csv_dir is not empty (i.e. not in current dir)
        os.makedirs(csv_dir, exist_ok=True)
        print(f"Created directory '{csv_dir}' for the dataset (if it didn't exist).")

    # Check if the CSV file itself exists
    if not os.path.exists(csv_path_to_use):
        print(f"CRITICAL ERROR: Dataset file not found at '{csv_path_to_use}'.")
        print("Please ensure the file exists at this path or update the 'csv_path_to_use' variable in the script.")
        print("Aborting script.")
    else:
        g_test_results_df = run_analysis(csv_path_to_use)

        # Generate LaTeX table for G-test results
        if g_test_results_df is not None and not g_test_results_df.empty:
            latex_table_df = g_test_results_df.head(15).copy()  # Use .copy() to avoid SettingWithCopyWarning

            # Format p-values for better readability in LaTeX
            latex_table_df['p_value_formatted'] = latex_table_df['p_value'].apply(
                lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"
            )
            latex_table_df['g_statistic'] = latex_table_df['g_statistic'].round(2)

            # Select and reorder columns for the LaTeX table
            latex_table_for_print = latex_table_df[
                ['word', 'spam_freq', 'ham_freq', 'g_statistic', 'p_value_formatted']]

            print("\nLaTeX table for G-test results (top 15):")
            print(latex_table_for_print.to_latex(index=False, escape=False, column_format='lrrrc',
                                                 header=['Word', 'Freq (Spam)', 'Freq (Ham)', 'G-statistic',
                                                         '$p$-value'],
                                                 caption='Top 15 words with significantly different frequencies between spam and ham emails (G-test).',
                                                 label='tab:gtest_results'))
        else:
            print(
                "\nNo G-test results to generate LaTeX table (possibly due to empty data, one class missing, or no significant words).")