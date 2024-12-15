import os
import pandas as pd
import random
import argparse

def load_dataset(file_path, text_column):
    """
    Load the dataset from a CSV file and extract the specified text column.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text data.

    Returns:
        List[str]: List of text entries.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.\n")
        return []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}\n")
        return []

    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in '{file_path}'.\n")
        return []

    texts = df[text_column].dropna().tolist()
    if not texts:
        print(f"Warning: No data found in column '{text_column}' of '{file_path}'.\n")
    return texts

def display_random_samples(texts, num_samples=20):
    """
    Display a specified number of random samples from the texts.

    Args:
        texts (List[str]): List of text entries.
        num_samples (int): Number of samples to display.
    """
    if not texts:
        print("No texts available to display.\n")
        return

    samples = random.sample(texts, min(num_samples, len(texts)))
    for idx, sample in enumerate(samples, 1):
        print(f"Sample {idx}:\n{sample}\n{'-'*80}")

def main():
    """
    Main function to evaluate pronoun resolution by displaying random samples.
    """
    parser = argparse.ArgumentParser(description="Evaluate Pronoun Resolution by Displaying Random Samples")
    args = parser.parse_args()

    # Define dataset configurations
    datasets = {
        "1": {
            "name": "IMDB Original",
            "file_path": "data/IMDB_Dataset.csv",
            "text_column": "review"
        },
        "2": {
            "name": "IMDB Resolved",
            "file_path": "data/IMDB_Dataset_Resolved.csv",
            "text_column": "resolved_review"
        },
        "3": {
            "name": "20 Newsgroups Original",
            "file_path": "data/20_Newsgroups/dataset.csv",
            "text_column": "text"
        },
        "4": {
            "name": "20 Newsgroups Resolved",
            "file_path": "data/20_Newsgroups/dataset_resolved.csv",
            "text_column": "resolved_text"
        },
        "5": {
            "name": "BBC News Classification Original",
            "file_path": "data/BBC-Dataset-News-Classification/dataset/dataset.csv",
            "text_column": "news"
        },
        "6": {
            "name": "BBC News Classification Resolved",
            "file_path": "data/BBC-Dataset-News-Classification/dataset/dataset_resolved.csv",
            "text_column": "resolved_news"
        }
    }

    # Menu Options
    menu = """
    ===== Pronoun Resolution Evaluation =====

    Select an option to view 20 random samples:

    1. IMDB Original Dataset
    2. IMDB Resolved Dataset
    3. 20 Newsgroups Original Dataset
    4. 20 Newsgroups Resolved Dataset
    5. BBC News Classification Original Dataset
    6. BBC News Classification Resolved Dataset
    7. Exit

    ==========================================
    """

    while True:
        print(menu)
        choice = input("Enter your choice (1-7): ").strip()

        if choice == "7":
            print("Exiting the evaluation tool. Goodbye!")
            break

        dataset = datasets.get(choice)
        if not dataset:
            print("Invalid choice. Please select a valid option.\n")
            continue

        print(f"\n--- Displaying 20 Random Samples from {dataset['name']} ---\n")
        texts = load_dataset(dataset["file_path"], dataset["text_column"])
        display_random_samples(texts, num_samples=20)
        print("\n--- End of Samples ---\n")

if __name__ == "__main__":
    main()
