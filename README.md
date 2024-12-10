Requirements
Python: Version 3.7 or higher.
Required Libraries:
nltk
scikit-learn
joblib
Environment: The program is compatible with Windows, macOS, and Linux.

Setup Instructions
1. Extract the ZIP File
Unzip the provided file to a desired location. Ensure all files (listed above) are extracted into the same directory.

2. Install Required Libraries
Run the following commands to install the required Python libraries:
pip install nltk scikit-learn joblib
3. Download NLTK Resources
The program uses NLTK for tokenization and stopword removal. Download the necessary resources by running the following commands in a Python shell:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
How to Run the Program
1. Training the Models
To train or retrain models, follow these steps:

Open a terminal or command prompt.
Navigate to the program directory:

cd /path/to/unzipped/folder
Run the program:

python cs5322f24.py
Choose option 1 or 2 (both train models):
Option 1: Train models for the first time.
Option 2: Retrain the models using the existing training files.
2. Testing with a File
To classify test sentences:

Prepare a test file (e.g., overtime_test.txt) with one sentence per line.
Run the program and choose option 3:

python cs7322f24.py
Enter the word ("overtime," "rubbish," or "tissue") and the test file name (e.g., overtime_test.txt).
The program will generate a result file named result_<word>_Scott_Mackenzie.txt (e.g., result_overtime_Scott_Mackenzie.txt) in the same directory.
3. Exit the Program
To exit, choose option 4.
