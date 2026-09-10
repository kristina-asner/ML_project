# Sentiment Analysis on IMDb Movie Reviews — Multinomial Naive Bayes with TF-IDF

An end to end Natural Language Processing (NLP) pipeline for binary sentiment classification on the IMDb Movie Reviews dataset. This project features a custom **from-scratch NumPy implementation of Multinomial Naive Bayes** with additive smoothing and numerically stable log space inference, combined with optimized TF-IDF n-gram vectorization.

---

## 👥 Authors
* **Kristina Asner**
* **May Ashkenazi**

---

## 📌 Project Overview & Problem Definition
* **Task:** Binary Sentiment Classification (Supervised Learning).
* **Objective:** Given raw text from movie reviews, predict whether the sentiment expressed is **Positive** (`label = 1`) or **Negative** (`label = 0`).
* **Dataset:** IMDb Movie Reviews dataset, pre split into:
  * `Train.csv`: 40,000 samples (50.05% Negative, 49.95% Positive) - perfectly balanced.
  * `Test.csv`: 5,000 held-out samples (49.90% Negative, 50.10% Positive) - strictly untouched during preprocessing and model selection to prevent data leakage.

---

## 🏗 Pipeline & Architectural Strategy

### 1. Text Preprocessing & Cleaning
* **HTML Stripping:** Cleans markup noise such as `<br />` breaks left from web scraping.
* **Regex Noise Removal:** Strips non-alphabetic characters and punctuation (`[^a-zA-Z\s]`).
* **Normalization (Case Folding):** Standardizes all words to lowercase.
* **Whitespace Normalization:** Compresses multi spaces and trims tokens.

### 2. Feature Engineering (TF-IDF Vectorization)
* **Zipf's Law Mitigation:** Raw word counts favor frequent non discriminative terms. TF-IDF applies an Inverse Document Frequency penalty:
  $$\text{tfidf}(t, d) = \text{tf}(t, d) \times \left(\ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1\right)$$
* **N-Gram Range:** `ngram_range=(1, 2)` (Unigrams + Bigrams) to capture local contextual sentiment (e.g., distinguishing `"not good"` from `"good"`).
* **Vocabulary Management:**
  * `stop_words="english"`: Eliminates uninformative syntactic function words.
  * `max_features=30,000`: Caps dimensionality to the top discriminative terms.
  * `min_df=5`: Prunes ultra-rare typo artifacts.
  * `sublinear_tf=True`: Dampens frequency scaling using $1 + \ln(\text{tf})$.
  * $L_2$ document vector normalization.
* **Leakage Prevention:** The `TfidfVectorizer` is **fit exclusively on `Train.csv`**, followed by transforming `Test.csv`.

---

## 🧠 Model Architecture & Mathematical Foundations

The core model is implemented **from scratch using NumPy and `scipy.sparse`**, without using `sklearn.naive_bayes`.

### 1. Generative Decision Rule
Multinomial Naive Bayes operates on the conditional independence assumption:
$$P(y \mid x) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

### 2. Additive (Laplace / Lidstone) Smoothing
To eliminate the Zero Frequency problem (where an unseen test term zeroes out the entire class probability), we apply additive smoothing:
$$\hat{P}(x_i \mid y) = \frac{\text{count}(x_i, y) + \alpha}{\sum_{j} \text{count}(x_j, y) + \alpha |V|}$$
Where:
* $\alpha = 1.0$: Laplace smoothing.
* $0 < \alpha < 1$: Lidstone smoothing (ideal for large sparse vocabularies).
* $\alpha \to 0$: Standard Maximum Likelihood Estimation (MLE).

### 3. Numerically Stable Log-Space Inference
To prevent floating-point underflow when multiplying thousands of small likelihoods, predictions are computed in log space:
$$\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{n} x_i \log P(x_i \mid y) \right]$$

Posterior class probabilities are derived via the **Log-Sum-Exp** trick:
$$\log P(y \mid x) = \text{jll}(x, y) - \left( m + \ln \sum_k e^{\text{jll}(x, k) - m} \right), \quad m = \max_k \text{jll}(x, k)$$

---

## ⚙️ Hyperparameter Tuning & Cross-Validation

Hyperparameter tuning was conducted exclusively on `Train.csv` using custom **3-Fold Stratified Cross-Validation**:

| Configuration | $\alpha$ Value | Mean CV $F_1$ Score | Std Dev | Wall-clock Time |
| :--- | :---: | :---: | :---: | :---: |
| **Laplace (Selected)** | **$1.0$** | **0.8733** | **±0.0017** | **~0.17s** |
| Lidstone | $0.1$ | 0.8731 | ±0.0014 | ~0.18s |
| Near-MLE | $10^{-9}$ | 0.8662 | ±0.0006 | ~0.17s |

> **Insight:** Laplace and Lidstone smoothing are statistically tied and both decisively outperform the unsmoothed MLE model, confirming the necessity of probability smoothing in sparse text classification.

---

## 📊 Evaluation & Results (Held-Out Test Set)

The final model trained on all 40,000 training examples was evaluated against the 5,000 unseen test reviews:

* **Overall Accuracy:** **88.0%**
* **Positive Class $F_1$-Score:** **0.8774**
* **Macro Avg $F_1$:** **0.88**

### Detailed Classification Report