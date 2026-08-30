# 🎬 IMDb Movie Reviews Sentiment Analysis

An end-to-end Natural Language Processing (NLP) pipeline for binary sentiment classification on the IMDb Movie Reviews dataset using **TF-IDF Vectorization** and **Multinomial Naive Bayes**.

---

## 📌 Project Overview

This project implements a complete binary sentiment classification workflow to predict whether a movie review expresses **Positive (`1`)** or **Negative (`0`)** sentiment. By pairing thorough text cleaning with TF-IDF feature extraction and additive smoothing, the project demonstrates how generative probabilistic models achieve strong accuracy and computational efficiency on sparse, high-dimensional text data.

### 👥 Authors
* **Kristina Asner** (ID: `212692255`)
* **May Ashkenazi** (ID: `212656631`)

---

## 📊 Dataset

The model is evaluated using the IMDb Movie Reviews benchmark split into two distinct subsets:
* **`Train.csv` (40,000 samples):** Used for noise cleaning, vocabulary fitting, TF-IDF transformation, and 3-fold cross-validation.
* **`Test.csv` (5,000 samples):** Held-out evaluation set kept strictly untouched until final inference to guarantee zero data leakage.

| Split | Total Samples | Class 0 (Negative) | Class 1 (Positive) |
|---|:---:|:---:|:---:|
| **Train** | 40,000 | 20,019 (50.05%) | 19,981 (49.95%) |
| **Test** | 5,000 | 2,495 (49.90%) | 2,505 (50.10%) |

---

## ⚙️ Pipeline Architecture & Methodology

### 1. Text Preprocessing & Cleaning
* **Noise Removal:** Strips HTML line breaks (e.g., `<br />`), digits, and special characters using regular expressions.
* **Case Folding & Normalization:** Converts all text to lowercase and removes standard English stopwords.

### 2. Feature Engineering (TF-IDF)
To address **Zipf's Law** (where frequent terms dominate and rare terms carry high discriminative value), raw counts are weighted via TF-IDF:
$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \left( \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1 \right)$$
* **N-gram Range:** Includes both unigrams and bigrams (`(1, 2)`) to capture local negation patterns (e.g., `"not good"`).
* **Vocabulary Constraints:** Capped at `max_features=30,000` with `min_df=5` and `sublinear_tf=True`.

### 3. Generative Modeling & Smoothing
* **Naive Independence:** Formulates the decision boundary via Bayes' Rule in log-space to ensure numerical stability and prevent floating-point underflow:
  $$\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{n} x_i \log P(x_i \mid y) \right]$$
* **Zero-Frequency Correction:** Applies additive smoothing parameter $\alpha$ to prevent unseen test words from zeroing out the posterior probability.

### 4. Hyperparameter Optimization
Evaluated across three smoothing regimes using **3-Fold Stratified Cross-Validation** on the training set:

| Configuration | Smoothing Strategy | Mean CV F1 | Std CV F1 | Fit Time |
|---|---|:---:|:---:|:---:|
| **`alpha = 0.1`** | **Lidstone Smoothing (Optimal)** | **`0.8754`** | **`0.0021`** | **`1.3s`** |
| `alpha = 1.0` | Laplace ("Add-One") | `0.8748` | `0.0026` | `1.3s` |
| `alpha ≈ 0` (`1e-9`) | Maximum Likelihood Estimation (MLE) | `0.8676` | `0.0026` | `1.8s` |

---

## 📈 Final Model Evaluation

The final classifier was trained on the entire training set with optimal parameter $\alpha = 0.1$ and evaluated on the 5,000 held-out test reviews:

* **Test-Set Binary F1-Score (Positive Class):** **`0.8777`**
* **Overall Test Accuracy:** **`88%`**