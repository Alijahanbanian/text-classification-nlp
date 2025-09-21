# Intent Classification with Machine Learning

This project implements a text classification pipeline to detect user intents from text queries (e.g., "how long should I cook steak for" → "cook_time"). It uses scikit-learn models: Logistic Regression, Multinomial Naive Bayes, and Linear SVM. The dataset consists of labeled JSON lines files for training, validation, and testing.

## Features

- Text preprocessing using NLTK (stopwords removal, lemmatization).
- TF-IDF vectorization for feature extraction.
- Model training and evaluation on a validation set to select the best model.
- Final evaluation on a test set with a detailed classification report saved as CSV.

## Dataset

The dataset contains approximately 15,000 training examples, 3,000 validation examples, and 7,500 test examples across 150 intent labels (e.g., "balance", "weather", "book_flight"). Data is provided in JSON Lines format.

- **train.json**: Training data.
- **validation.json**: Validation data.
- **test.json**: Test data.

## Prerequisites

To run this project, you need:

- Python 3.10+ (tested on 3.10.11).
- Jupyter Notebook or JupyterLab to run `main.ipynb`.
- Git to clone the repository.
- Internet connection for downloading NLTK data.

## Installation

Follow these steps to set up the project environment and download all dependencies:

1. **Clone the Repository**

2. **Install Python Dependencies**:

   - Install required packages listed in `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```
   - The `requirements.txt` includes:
     - pandas==2.2.2
     - scikit-learn==1.5.1
     - nltk==3.8.1

3. **Download NLTK Data**:
   - The project requires specific NLTK datasets for text preprocessing.
   - Run the following in a Python console or add to the notebook:
     ```
     import nltk
     nltk.download('stopwords')
     nltk.download('wordnet')
     nltk.download('omw-1.4')
     ```
   - These datasets are downloaded once and stored locally (e.g., in `~/nltk_data`).

## How to Run

1. Ensure all dependencies and NLTK data are installed (see Installation).
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab:
   ```
   jupyter notebook main.ipynb
   ```
3. Run all cells in the notebook:
   - Loads data from `./Datasets/`.
   - Preprocesses text, applies TF-IDF vectorization, trains models, and evaluates them.
   - Selects the best model (Logistic Regression) based on validation accuracy.
   - Generates a test set evaluation report saved as `outputs/test_classification_report.csv`.

## Project Structure

- `main.ipynb`: Core Jupyter notebook with the end-to-end pipeline.
- `Datasets/`:
  - `train.json`: Training dataset.
  - `validation.json`: Validation dataset.
  - `test.json`: Test dataset.
- `outputs/`:
  - `test_classification_report.csv`: Classification metrics for the test set.
- `requirements.txt`: List of Python dependencies.

## Results

- **Best Model**: Logistic Regression
- **Validation Accuracy**: 0.848
- **Test Accuracy**: 0.8484
- **Macro Avg F1-Score**: 0.8485

Selected metrics from `outputs/test_classification_report.csv`:

| Label               | Precision | Recall | F1-Score | Support |
| ------------------- | --------- | ------ | -------- | ------- |
| accept_reservations | 0.69      | 0.94   | 0.80     | 50      |
| account_blocked     | 0.81      | 0.86   | 0.83     | 50      |
| alarm               | 0.96      | 0.98   | 0.97     | 50      |
| ... (full in CSV)   | ...       | ...    | ...      | ...     |
| **Macro Avg**       | 0.86      | 0.85   | 0.85     | 7500    |

See `outputs/test_classification_report.csv` for full per-label metrics.

## Limitations and Future Work

- Uses traditional ML models; deep learning models (e.g., BERT) could improve accuracy.
- Dataset is balanced but may not capture all real-world query variations.
- No hyperparameter tuning; adding GridSearchCV could enhance model performance.
- Limited to English text; multilingual support could be added.

## Troubleshooting

- **NLTK data issues**: Ensure `nltk.download()` commands are run successfully.
- **File not found**: Verify `Datasets/` contains `train.json`, `validation.json`, and `test.json`.
- **Dependency errors**: Check that Python 3.10+ is used and `requirements.txt` packages are installed.
- For other issues, open a GitHub issue at `https://github.com/YOUR_USERNAME/intent-classification-ml`.
