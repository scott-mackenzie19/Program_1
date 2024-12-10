import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Function to load training data from a file
def load_data(file_path):
    """
    Loads training data from a text file and separates sentences into two categories (senses).
    Args:
        file_path (str): Path to the file containing training data.
    Returns:
        list, list: Sentences for sense 1 and sense 2.
    """
    sense1, sense2 = [], []
    with open(file_path, "r", encoding="utf-8") as file:  # Read file with UTF-8 encoding
        lines = file.readlines()
        flag = 0  # Tracks whether sentences belong to sense 1 or sense 2
        for line in lines:
            if line.strip() == "1":  # Mark beginning of sense 1 sentences
                flag = 1
            elif line.strip() == "2":  # Mark beginning of sense 2 sentences
                flag = 2
            elif flag == 1 and line.strip():  # Add to sense 1
                sense1.append(line.strip())
            elif flag == 2 and line.strip():  # Add to sense 2
                sense2.append(line.strip())
    return sense1, sense2

# Function to train and save a machine learning model
def train_model(sense1_sentences, sense2_sentences, output_model_path):
    """
    Trains a Naive Bayes model using the provided sentences and saves it to a file.
    Args:
        sense1_sentences (list): Sentences belonging to sense 1.
        sense2_sentences (list): Sentences belonging to sense 2.
        output_model_path (str): Path to save the trained model.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Combine sentences and assign labels
    sentences = sense1_sentences + sense2_sentences
    labels = [1] * len(sense1_sentences) + [2] * len(sense2_sentences)
    
    # Define a pipeline for text processing and classification
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words=stopwords.words('english'))),
        ('classifier', MultinomialNB())
    ])
    
    # Train the model
    pipeline.fit(sentences, labels)
    joblib.dump(pipeline, output_model_path)  # Save the trained model to a file
    print(f"Model trained and saved to {output_model_path}")

# Function to classify sentences using a trained model
def classify_sentences(sentences, model_path):
    """
    Classifies sentences using a pre-trained model.
    Args:
        sentences (list): List of sentences to classify.
        model_path (str): Path to the pre-trained model.
    Returns:
        list: Predicted labels for the sentences.
    """
    model = joblib.load(model_path)  # Load the trained model
    predictions = model.predict(sentences)  # Predict the labels
    return predictions

# Word-specific model file paths
MODEL_PATH_OVERTIME = "wsd_overtime.pkl"
MODEL_PATH_RUBBISH = "wsd_rubbish.pkl"
MODEL_PATH_TISSUE = "wsd_tissue.pkl"

# Function to process a test file and save results automatically
def process_test_file_auto(test_file, word):
    """
    Processes a test file, classifies sentences, and saves results to an auto-generated file.
    Args:
        test_file (str): Path to the test file.
        word (str): Word being classified (overtime, rubbish, or tissue).
    """
    model_path = f"wsd_{word}.pkl"  # Determine model path
    result_file = f"result_{word}_Scott_Mackenzie.txt"  # Auto-generate result file name
    
    with open(test_file, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.readlines() if line.strip()]
    
    predictions = classify_sentences(sentences, model_path)  # Get predictions
    
    with open(result_file, "w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")
    
    print(f"Results saved to {result_file}")

# Main program
if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Train models from text files")
        print("2. Train models again (retrain models)")
        print("3. Process a test file (auto-save results)")
        print("4. Exit")
        mode = input("Enter your choice: ").strip()
        
        if mode in ["1", "2"]:
            # Train models
            print("Training models...")
            sense1, sense2 = load_data("overtime.txt")
            train_model(sense1, sense2, MODEL_PATH_OVERTIME)

            sense1, sense2 = load_data("rubbish.txt")
            train_model(sense1, sense2, MODEL_PATH_RUBBISH)

            sense1, sense2 = load_data("tissue.txt")
            train_model(sense1, sense2, MODEL_PATH_TISSUE)
            print("All models trained and saved.")
        elif mode == "3":
            # Process a test file with auto-save
            word = input("Enter the word (overtime/rubbish/tissue): ").strip().lower()
            test_file = input("Enter the name of the test file: ").strip()
            
            if word in ["overtime", "rubbish", "tissue"]:
                process_test_file_auto(test_file, word)
            else:
                print("Invalid word. Please choose from overtime, rubbish, or tissue.")
        elif mode == "4":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
