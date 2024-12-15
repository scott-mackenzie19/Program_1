# train_classifier.py

import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import warnings
from transformers.utils import logging

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", message="Some weights of BertForSequenceClassification were not initialized")
logging.set_verbosity_error()

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_and_prepare_data(file_path, tokenizer, test_size=0.2):
    """
    Load dataset, encode labels, and split into training and testing sets.

    Args:
        file_path (str): Path to the CSV file.
        tokenizer (BertTokenizer): Pretrained BERT tokenizer.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        train_dataset (IMDBDataset): Training dataset.
        test_dataset (IMDBDataset): Testing dataset.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if 'sentiment' not in df.columns or 'review' not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")
    
    # Encode labels: positive -> 1, negative -> 0
    label_mapping = {'positive': 1, 'negative': 0}
    df = df[df['sentiment'].isin(label_mapping.keys())]
    df['label'] = df['sentiment'].map(label_mapping)
    
    texts = df['review'].tolist()
    labels = df['label'].tolist()
    
    # Split into training and testing
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)
    
    return train_dataset, test_dataset

def compute_metrics(pred):
    """
    Compute evaluation metrics.

    Args:
        pred: Predictions and labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1-score.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune_bert(train_dataset, test_dataset, output_dir, model_name='bert-base-uncased', epochs=3, batch_size=16):
    """
    Fine-tune BERT model using HuggingFace's Trainer API.

    Args:
        train_dataset (IMDBDataset): Training dataset.
        test_dataset (IMDBDataset): Testing dataset.
        output_dir (str): Directory to save the model.
        model_name (str): Pretrained BERT model name.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        Trainer: Trained Trainer object.
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',  # Updated from 'evaluation_strategy' to 'eval_strategy'
        save_strategy='epoch',
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=train_dataset.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir)
    
    return trainer

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune BERT for Sentiment Detection on IMDB Datasets")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model name (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Define datasets
    datasets = {
        "1": {
            "name": "IMDB Original Dataset",
            "file_path": "data/IMDB_Dataset.csv",
            "output_dir": "models/bert_imdb_original"
        },
        "2": {
            "name": "IMDB Resolved Dataset",
            "file_path": "data/IMDB_Dataset_Resolved.csv",
            "output_dir": "models/bert_imdb_resolved"
        }
    }
    
    # Menu Options
    menu = """
    ===== BERT Sentiment Classifier Training =====

    Select an option to train on a dataset:

    1. Train on IMDB Original Dataset
    2. Train on IMDB Resolved Dataset
    3. Exit

    =============================================
    """
    
    while True:
        print(menu)
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "3":
            print("Exiting the training tool. Goodbye!")
            break
        
        dataset = datasets.get(choice)
        if not dataset:
            print("Invalid choice. Please select a valid option.\n")
            continue
        
        print(f"\n--- Training on {dataset['name']} ---\n")
        
        try:
            train_dataset, test_dataset = load_and_prepare_data(
                file_path=dataset['file_path'],
                tokenizer=tokenizer
            )
        except Exception as e:
            print(f"Error loading data: {e}\n")
            continue
        
        # Create output directory if it doesn't exist
        os.makedirs(dataset['output_dir'], exist_ok=True)
        
        print(f"Fine-tuning BERT model on {dataset['name']}...\n")
        trainer = fine_tune_bert(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=dataset['output_dir'],
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print(f"\nTraining completed. Model saved to '{dataset['output_dir']}'\n")
    
    print("All selected trainings have been completed.")

if __name__ == "__main__":
    main()
