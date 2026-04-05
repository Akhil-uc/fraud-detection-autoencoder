# Fraud Detection using AutoEncoder

## Overview

This project implements an **unsupervised anomaly detection system** to identify fraudulent credit card transactions using an AutoEncoder model. The approach is particularly effective for highly imbalanced datasets where fraudulent cases are rare.

---

## Features

* End-to-end ML pipeline
* Data preprocessing and scaling
* AutoEncoder-based anomaly detection (PyOD)
* Evaluation using classification metrics
* Visualization of anomaly scores
* Modular and clean project structure

---

## Project Structure

```
fraud_detection/
│
├── data/
│   └── fraud.csv
│
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

* Source: Kaggle Credit Card Fraud Dataset
* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Highly imbalanced dataset

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python -m src.main
```

---

## Methodology

1. Load dataset
2. Preprocess data using StandardScaler
3. Train AutoEncoder on normal transactions only
4. Predict anomaly scores
5. Classify anomalies
6. Evaluate performance
7. Visualize anomaly distribution

---

## Results

**Accuracy:** 98%

### Classification Report

```
precision    recall  f1-score   support

0       1.00      0.98      0.99    284315
1       0.07      0.84      0.13       492
```

### Confusion Matrix

```
[[278629   5686]
 [    77    415]]
```

### Key Insights

* High recall (0.84) → most frauds detected
* Low precision (0.07) → many false positives

---

## Visualization

The anomaly score distribution shows a right-skewed curve:

* Most transactions → low scores (normal)
* Few transactions → high scores (fraud)

---

## Limitations

* High false positive rate
* Overlapping anomaly scores
* Not optimized threshold

---

## Future Improvements

* Threshold tuning
* Use Isolation Forest / XGBoost
* Feature engineering
* Hyperparameter tuning

---

## Requirements

```
pyod
torch
pandas
numpy
scikit-learn
matplotlib
```



---

## ⭐ If you found this useful, consider giving a star!
