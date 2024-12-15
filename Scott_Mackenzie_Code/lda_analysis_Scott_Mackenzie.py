import os
import pandas as pd
import re
from gensim import corpora, models
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    """
    Preprocess the input text by removing non-alphabetic characters,
    tokenizing, converting to lowercase, and removing stopwords.
    """
    # Remove non-alphabetic characters and tokenize
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

def preprocess_20_newsgroups(data_dir, list_csv_path):
    """
    Preprocess the 20 Newsgroups dataset.
    
    Parameters:
    - data_dir: Directory containing the 20 newsgroup files.
    - list_csv_path: Path to 'list.csv' mapping document_id to newsgroups.
    
    Returns:
    - List of preprocessed documents.
    """
    # Load the list.csv to map document_id to newsgroup
    list_df = pd.read_csv(list_csv_path)
    doc_id_to_group = dict(zip(list_df['document_id'], list_df['newsgroup']))
    
    processed_docs = []
    
    # Iterate through each newsgroup file
    for group in list_df['newsgroup'].unique():
        group_file = os.path.join(data_dir, f"{group}.txt")
        if not os.path.isfile(group_file):
            print(f"Warning: {group_file} does not exist. Skipping.")
            continue
        with open(group_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Split the content into individual messages based on headers
            messages = re.split(r'(?=Newsgroup:)', content)
            for message in messages:
                # Remove headers
                message_body = re.sub(r'^Newsgroup:.*\nDocument_id:.*\nFrom:.*\nSubject:.*\n', '', message, flags=re.MULTILINE)
                if message_body.strip():
                    tokens = preprocess(message_body)
                    processed_docs.append(tokens)
    
    return processed_docs

def preprocess_bbc_news(data_dir, dataset_csv_path):
    """
    Preprocess the BBC News Classification dataset.
    
    Parameters:
    - data_dir: Directory containing 'dataset.csv'.
    - dataset_csv_path: Path to 'dataset.csv'.
    
    Returns:
    - List of preprocessed news articles.
    """
    # Load the dataset.csv
    df = pd.read_csv(dataset_csv_path)
    documents = df['news'].tolist()
    processed = [preprocess(doc) for doc in documents]
    return processed

def run_lda(processed_docs, num_topics=20):
    """
    Perform LDA topic modeling and calculate topic coherence.
    
    Parameters:
    - processed_docs: List of tokenized and preprocessed documents.
    - num_topics: Number of topics for LDA.
    
    Returns:
    - coherence score.
    """
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(text) for text in processed_docs]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence

def main():
    # Paths for 20 Newsgroups
    newsgroups_data_dir = '20_Newsgroups'  # Replace with your actual directory
    newsgroups_list_csv = os.path.join(newsgroups_data_dir, 'list.csv')
    
    # Paths for BBC News Classification
    bbc_dataset_dir = 'BBC-Dataset-News-Classification'
    bbc_dataset_csv = os.path.join(bbc_dataset_dir, 'dataset', 'dataset.csv')
    
    # 1. Process 20 Newsgroups - Unmodified
    print("Processing 20 Newsgroups - Unmodified...")
    processed_unmod_20ng = preprocess_20_newsgroups(newsgroups_data_dir, newsgroups_list_csv)
    coherence_unmod_20ng = run_lda(processed_unmod_20ng, num_topics=20)
    print(f"20 Newsgroups - Unmodified Topic Coherence: {coherence_unmod_20ng:.4f}")
    
    # 2. Process 20 Newsgroups - Resolved
    print("\nProcessing 20 Newsgroups - Resolved...")
    resolved_newsgroups_csv = os.path.join(newsgroups_data_dir, '20_newsgroups_resolved.csv')  # Adjust if different
    if os.path.isfile(resolved_newsgroups_csv):
        # Assuming resolved data is processed similarly
        processed_res_20ng = preprocess_20_newsgroups(newsgroups_data_dir, resolved_newsgroups_csv)
        coherence_res_20ng = run_lda(processed_res_20ng, num_topics=20)
        print(f"20 Newsgroups - Resolved Topic Coherence: {coherence_res_20ng:.4f}")
    else:
        print("Resolved 20 Newsgroups dataset not found. Skipping resolved analysis for 20 Newsgroups.")
    
    # 3. Process BBC News Classification - Unmodified
    print("\nProcessing BBC News Classification - Unmodified...")
    processed_unmod_bbc = preprocess_bbc_news(bbc_dataset_dir, bbc_dataset_csv)
    coherence_unmod_bbc = run_lda(processed_unmod_bbc, num_topics=20)
    print(f"BBC News - Unmodified Topic Coherence: {coherence_unmod_bbc:.4f}")
    
    # 4. Process BBC News Classification - Resolved
    print("\nProcessing BBC News Classification - Resolved...")
    resolved_bbc_csv = os.path.join(bbc_dataset_dir, 'dataset', 'BBC_Dataset_Resolved.csv')  # Adjust if different
    if os.path.isfile(resolved_bbc_csv):
        processed_res_bbc = preprocess_bbc_news(bbc_dataset_dir, resolved_bbc_csv)
        coherence_res_bbc = run_lda(processed_res_bbc, num_topics=20)
        print(f"BBC News - Resolved Topic Coherence: {coherence_res_bbc:.4f}")
    else:
        print("Resolved BBC News dataset not found. Skipping resolved analysis for BBC News.")

if __name__ == "__main__":
    main()
