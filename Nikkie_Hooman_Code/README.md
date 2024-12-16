# Sentiment Analysis with Coreference Resolution and Topic Modeling

## Overview

This research focalizes on the influence of pronominal resolution (coreferential resolution) on sentiment models. The change of the text data in the treatment of the BERT and the different modeling of the topic are also investigated.

The complete procedure consists of:
- The simple rule-based function examines the coreference resolution task.
- Embodying and judging of sentiment analysis models using various data groups.
- Manual scrutiny of coreference resolution.
- Topic modeling using Latent Dirichlet Allocation (LDA).

---

## File Descriptions

- **`IMDB_Dataset.csv`**: The information provided in this dataset is about the movie reviews and their respective labels of the sentiment.
- **`sentiment_analysis_with_coref.py`**: A Python script that serves as a tool to automate the taskstep
- **`results/`**: Store the training outputs in this particular folder.
- **`logs/`**: This folder is where the training logs are located.

---

## Workflow

### Step 1: Coreference Resolution

A simple rule-based function is used to allocate the pronouns to the items they refer to in an entity. It checks the appearance of the pronoun in the text and then replaces it with the appropriate antecedent for it. This is used on the entire data set and the resulting entry is a new column called resolved_text.

### Step 2: Data Loading and Preprocessing

- IMDB dataset is loaded.
- Utilize the coreference resolution function to remove coreferences.
- The dataset is split into training and test sets.

### Step 3: Sentiment Analysis

#### Datasets

Four groups of datasets are created:
1. **Unmodified Training & Testing**
2. **Resolved Training & Testing**
3. **Resolved Training & Unmodified Testing**
4. **Unmodified Training & Resolved Testing**

#### Model

A BERT-based classifier (HuggingFace transformers library) is fine-tuned for binary sentiment classification.

#### Evaluation

Each dataset group is trained and evaluated using Trainer from HuggingFace. Results are logged and compared to analyze the impact of coreference resolution.

### Step 4: Manual Evaluation of Coreference Resolution

50 samples from the dataset are manually inspected to verify the accuracy of pronoun resolution.

### Step 5: Topic Modeling

Latent Dirichlet Allocation (LDA) is performed on:

1. Unmodified Text
2. Resolved Text

Topic terms are extracted for comparison.

---

## Results

### Sentiment Analysis

Results are provided for each dataset group, including evaluation loss, runtime, and other metrics. Performance differences highlight how pronoun resolution affects training and testing.

### Coreference Resolution

Manual inspection evaluates the quality of the resolved text.

### Topic Modeling

Topic distributions differ between unmodified and resolved datasets, revealing shifts in thematic structures due to pronoun resolution.

---

## Dependencies

- Python 3.8+
- Libraries:
  - `transformers`
  - `torch`
  - `pandas`
  - `scikit-learn`
  - `numpy`

---

## How to Run

1. Install the required dependencies:
   ```bash
   pip install transformers torch pandas scikit-learn numpy
   ```
2. Place the IMDB dataset in the same directory as the script.
3. Run the script:
   ```bash
   python sentiment_analysis_with_coref.py
   ```
4. Results will be saved in the `results/` directory.

---

## Key Takeaways

1. Pronoun resolution introduces variability in model performance, emphasizing the importance of preprocessing in NLP tasks.
2. Topic modeling shows distinct thematic changes between resolved and unmodified datasets.
3. Coreference resolution methods must be carefully evaluated for accuracy to ensure meaningful preprocessing.

---

## Future Work

- Implement advanced coreference resolution using libraries like `NeuralCoref` or HuggingFace models.
- Use larger datasets to validate findings.
- Explore the impact of pronoun resolution on other NLP tasks such as summarization or question answering.

