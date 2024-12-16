### Sentiment Analysis with Coreference Resolution and Topic Modeling

## Overview
This research focalizes on the influence of pronominal resolution (coreferential resolution) on sentiment models. The change of the text data in the treatment of the BERT and the different modeling of the topic are also investigated.
The complete procedure consists of:
The simple rule-based function examines the coreference resolution task.
Embodying and judging of sentiment analysis models using various data groups.
Manual scrutiny of coreference resolution.
Topic modeling using Latent Dirichlet Allocation (LDA).

File Descriptions
IMDB_Dataset.csv: The information provided in this dataset is about the movie reviews and their respective labels of the sentiment.
sentiment_analysis_with_coref.py: A Python script that serves as a tool to automate the taskstep
results/: Store the training outputs in this particular folder.
logs/: This folder is where the training logs are located.

Workflow
Step 1: Coreference Resolution
A simple rule-based function is used to allocate the pronouns to the items they refer to in an entity. It checks the appearance of the pronoun in the text and then replaces it with the appropriate antecedent for it. This is used on the entire data set and the resulting entry is a new column called resolved_text.
Step 2: Data Loading and Preprocessing
IMDB dataset is loaded.
Utilize the coreference resolution function to remove coreferences.
The dataset is split into training and test sets.


Step 3: Sentiment Analysis
Datasets
Four groups of datasets are created:
Unmodified Training & Testing
Resolved Training & Testing
Resolved Training & Unmodified Testing
Unmodified Training & Resolved Testing

Model
A BERT-based classifier (HuggingFace transformers library) is fine-tuned for binary sentiment classification.
Evaluation
Each dataset group is trained and evaluated using Trainer from HuggingFace. Results are logged and compared to analyze the impact of coreference resolution.
Step 4: Manual Evaluation of Coreference Resolution
50 samples from the dataset are manually inspected to verify the accuracy of pronoun resolution.
Step 5: Topic Modeling
Latent Dirichlet Allocation (LDA) is performed on:
Unmodified Text
Resolved Text

Topic terms are extracted for comparison.

Results

Sentiment Analysis

Results are provided for each dataset group, including evaluation loss, runtime, and other metrics. Performance differences highlight how pronoun resolution affects training and testing.

Coreference Resolution

Manual inspection evaluates the quality of the resolved text.

Topic Modeling

Topic distributions differ between unmodified and resolved datasets, revealing shifts in thematic structures due to pronoun resolution.


