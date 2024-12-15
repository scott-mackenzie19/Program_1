import os
import pandas as pd
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import re
import argparse
from tqdm import tqdm

def load_model():
    """
    Load the coreference resolution model from AllenNLP.
    """
    print("Loading coreference resolution model...")
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    )
    print("Model loaded successfully.")
    return predictor

def resolve_pronouns(text, predictor):
    """
    Resolve pronouns in the given text using the provided predictor.

    Args:
        text (str): The input text with potential pronouns.
        predictor (Predictor): The AllenNLP coreference resolution predictor.

    Returns:
        str: Text with pronouns resolved.
    """
    try:
        resolved_text = predictor.coref_resolved(text)
        return resolved_text
    except Exception as e:
        print(f"Error resolving pronouns in text: {e}")
        return text  # Return original text if resolution fails

def process_imdb(input_path, output_path, predictor):
    """
    Apply pronoun resolution to the IMDB dataset.

    Args:
        input_path (str): Path to the input IMDB dataset CSV file.
        output_path (str): Path to save the resolved IMDB dataset CSV file.
        predictor (Predictor): The AllenNLP coreference resolution predictor.

    Returns:
        None
    """
    print("\nProcessing IMDB Dataset...")
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading '{input_path}': {e}")
        return

    if 'review' not in df.columns:
        print("Error: 'review' column not found in IMDB dataset.")
        return

    print("Applying pronoun resolution to IMDB reviews...")
    tqdm.pandas(desc="Resolving Pronouns")
    df['resolved_review'] = df['review'].progress_apply(lambda x: resolve_pronouns(x, predictor))

    try:
        df.to_csv(output_path, index=False)
        print(f"Resolved IMDB dataset saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving resolved IMDB dataset: {e}")

def preprocess_20_newsgroups_message(message):
    """
    Preprocess individual message text by removing headers.

    Args:
        message (str): The raw message text.

    Returns:
        str: Preprocessed message text.
    """
    # Remove headers
    message_body = re.sub(r'^Newsgroup:.*\nDocument_id:.*\nFrom:.*\nSubject:.*\n', '', message, flags=re.MULTILINE)
    message_body = message_body.strip()
    return message_body

def process_20_newsgroups(data_dir, list_csv_path, output_path, predictor):
    """
    Apply pronoun resolution to the 20 Newsgroups dataset.

    Args:
        data_dir (str): Directory containing the 20 newsgroup .txt files.
        list_csv_path (str): Path to 'list.csv' mapping document_id to newsgroups.
        output_path (str): Path to save the resolved 20 Newsgroups dataset CSV file.
        predictor (Predictor): The AllenNLP coreference resolution predictor.

    Returns:
        None
    """
    print("\nProcessing 20 Newsgroups Dataset...")
    if not os.path.isfile(list_csv_path):
        print(f"Error: List file '{list_csv_path}' does not exist.")
        return

    try:
        list_df = pd.read_csv(list_csv_path)
    except Exception as e:
        print(f"Error reading '{list_csv_path}': {e}")
        return

    doc_id_to_group = dict(zip(list_df['document_id'], list_df['newsgroup']))

    processed_data = {'document_id': [], 'newsgroup': [], 'resolved_text': []}

    for group in list_df['newsgroup'].unique():
        group_file = os.path.join(data_dir, f"{group}.txt")
        if not os.path.isfile(group_file):
            print(f"Warning: '{group_file}' does not exist. Skipping this newsgroup.")
            continue
        print(f"Processing newsgroup file: {group_file}")
        with open(group_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Split the content into individual messages based on headers
            messages = re.split(r'(?=Newsgroup:)', content)
            for message in messages:
                if message.strip() == '':
                    continue
                # Extract Document_id
                doc_id_match = re.search(r'Document_id:\s*(\d+)', message)
                if doc_id_match:
                    doc_id = doc_id_match.group(1)
                    newsgroup = doc_id_to_group.get(doc_id, 'unknown')
                else:
                    doc_id = 'unknown'
                    newsgroup = 'unknown'

                message_body = preprocess_20_newsgroups_message(message)
                if message_body:
                    resolved_text = resolve_pronouns(message_body, predictor)
                    processed_data['document_id'].append(doc_id)
                    processed_data['newsgroup'].append(newsgroup)
                    processed_data['resolved_text'].append(resolved_text)

    df_resolved = pd.DataFrame(processed_data)
    try:
        df_resolved.to_csv(output_path, index=False)
        print(f"Resolved 20 Newsgroups dataset saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving resolved 20 Newsgroups dataset: {e}")

def process_bbc_news(data_dir, dataset_csv_path, output_path, predictor):
    """
    Apply pronoun resolution to the BBC News Classification dataset.

    Args:
        data_dir (str): Directory containing 'dataset.csv'.
        dataset_csv_path (str): Path to 'dataset.csv'.
        output_path (str): Path to save the resolved BBC News dataset CSV file.
        predictor (Predictor): The AllenNLP coreference resolution predictor.

    Returns:
        None
    """
    print("\nProcessing BBC News Classification Dataset...")
    if not os.path.isfile(dataset_csv_path):
        print(f"Error: Dataset file '{dataset_csv_path}' does not exist.")
        return

    try:
        df = pd.read_csv(dataset_csv_path)
    except Exception as e:
        print(f"Error reading '{dataset_csv_path}': {e}")
        return

    if 'news' not in df.columns or 'type' not in df.columns:
        print("Error: 'news' and/or 'type' columns not found in BBC dataset.")
        return

    print("Applying pronoun resolution to BBC news articles...")
    tqdm.pandas(desc="Resolving Pronouns")
    df['resolved_news'] = df['news'].progress_apply(lambda x: resolve_pronouns(x, predictor))

    try:
        df.to_csv(output_path, index=False)
        print(f"Resolved BBC News dataset saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving resolved BBC News dataset: {e}")

def main():
    """
    Main function to perform pronoun resolution on multiple datasets.
    """
    parser = argparse.ArgumentParser(description="Pronoun Resolution for Multiple NLP Datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        choices=['imdb', '20newsgroups', 'bbc'],
        required=True,
        help="Datasets to process: 'imdb', '20newsgroups', 'bbc'"
    )
    args = parser.parse_args()

    # Load the coreference resolution model once
    predictor = load_model()

    for dataset in args.datasets:
        if dataset == 'imdb':
            input_path = 'data/IMDB_Dataset.csv'
            output_path = 'data/IMDB_Dataset_Resolved.csv'
            process_imdb(input_path, output_path, predictor)
        elif dataset == '20newsgroups':
            data_dir = 'data/20_Newsgroups'
            list_csv_path = os.path.join(data_dir, 'list.csv')
            output_path = 'data/20_Newsgroups_Resolved.csv'
            process_20_newsgroups(data_dir, list_csv_path, output_path, predictor)
        elif dataset == 'bbc':
            data_dir = 'data/BBC-Dataset-News-Classification/dataset'
            dataset_csv_path = os.path.join(data_dir, 'dataset.csv')
            output_path = 'data/BBC-Dataset-News-Classification/dataset/BBC_Dataset_Resolved.csv'
            process_bbc_news(data_dir, dataset_csv_path, output_path, predictor)
        else:
            print(f"Dataset '{dataset}' is not recognized. Skipping.")

if __name__ == "__main__":
    main()
