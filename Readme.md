# Sentiment Analysis on IMDb Reviews 

This project implements **sentiment classification** on the IMDb Movie Reviews dataset (50,000 labeled reviews).  
We benchmarked a **Machine Learning baseline (Logistic Regression with TF-IDF)** against a **Deep Learning model (BiLSTM with embeddings)**, with systematic preprocessing, feature engineering, and hyperparameter experiments.  

 **GitHub Repo**: [https://github.com/lscblack/Sentiment-Analysis](https://github.com/lscblack/Sentiment-Analysis)  

---

## Project Structure  

```

Sentiment-Analysis/
│── dataset/                 # Raw and preprocessed datasets (train/val/test CSVs)
│── notebooks/               # Jupyter notebooks for EDA, preprocessing, and experiments
│── saved_data/              # Persisted TF-IDF vectors, embeddings, and trained models
│── requirements.txt         # Dependencies for pip
│── environment.yml          # Conda environment specification
│── Readme.md                # Project documentation

````

---

## Installation & Setup  

You can install dependencies using **pip** or **conda**.  

### Option 1: Using pip  
```bash
git clone https://github.com/lscblack/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
````

### Option 2: Using conda (Recommended)

```bash
git clone https://github.com/lscblack/Sentiment-Analysis.git
cd Sentiment-Analysis
conda env create -f environment.yml
conda activate sentiment-analysis
```

---

##  Dataset Overview

* **Dataset**: IMDb Movie Reviews (50,000 labeled reviews, Kaggle)
* **Columns**: `review` (text), `sentiment` (positive/negative)
* **Class distribution**: Balanced (≈50/50)
* **Preprocessing**:

  * HTML tags removed
  * Text lowercased
  * Non-alphabetic symbols removed (apostrophes preserved)
  * Contraction expansion (`don't` → `do not`)
  * Stopword removal for selected tokenization strategies

### Data Splitting

* Training: 80% (~39,666 reviews)
* Validation: 10% (~4,958 reviews)
* Test: 10% (~4,958 reviews)

---

## 🔹 Preprocessing & Feature Engineering

* **Tokenization strategies**: Basic, Full, Strict
* **TF-IDF Features**: 4 configurations (unigrams_5k, unigrams_10k, bigrams_5k, bigrams_10k)
* **Deep Learning Features**:

  * Sequence length: 300 tokens
  * Vocabulary: 20,000 words
  * GloVe embeddings: 100-dimensional, 98.94% coverage

---

## Model Implementation

### Logistic Regression (ML Baseline)

* Features: TF-IDF bigrams, max_features=30,000
* Regularization: C=10.0
* Validation Accuracy: 0.897
* F1-score: 0.897

### BiLSTM with Embeddings (DL Model)

| exp_id | notes                        | trainable_embedding | lstm_units | batch | lr     | val_accuracy | val_f1_macro | val_logloss | best_epoch |
| ------ | ---------------------------- | ------------------- | ---------- | ----- | ------ | ------------ | ------------ | ----------- | ---------- |
| DL-A   | Random embeddings baseline   | ✅ True              | 128        | 64    | 0.0010 | 0.8246       | 0.8245       | 0.4066      | 5          |
| DL-C   | GloVe fine-tune, larger LSTM | ✅ True              | 256        | 32    | 0.0001 | 0.8145       | 0.8144       | 0.3917      | 9          |
| DL-B   | GloVe frozen embeddings      | ❌ False             | 128        | 64    | 0.0005 | 0.7268       | 0.7247       | 0.5329      | 9          |

---

##  Key Insights

* Logistic Regression slightly outperformed BiLSTM on this dataset (~90% vs ~89% accuracy).
* Deep Learning captures contextual patterns but requires more data and compute.
* Preprocessing decisions (tokenization, stopword removal, contraction expansion) improved model performance and vocabulary efficiency.
* Sequence length for LSTM (300 tokens) chosen based on the 95th percentile of review lengths.

---

##  How to Run

1. Open Jupyter Notebook:

```bash
jupyter notebook
```

2. Run notebooks in order:

   * `01_dataset_eda.ipynb` → EDA and visualizations
   * `02_preprocessing_features.ipynb` → Preprocessing & feature engineering
   * `03_model_training.ipynb` → Train ML & DL models
   * `04_experiments_results.ipynb` → Analyze results

---

##  Notes

* Models, embeddings, and TF-IDF vectors are cached under `saved_data/`.
* Random seeds are fixed for reproducibility.
* Future improvements: Experiment with Transformer-based models (BERT, RoBERTa) and systematic hyperparameter search.

---

##  References

* Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. ACL.
* Kaggle IMDb Dataset: [IMDb 50k Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP.


