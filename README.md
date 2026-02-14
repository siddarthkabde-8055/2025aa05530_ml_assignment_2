# ML Assignment 2 — Multi‑Class Classification (UCI Dry Bean Dataset)

## a. Problem statement

The objective of this assignment is to build and evaluate multiple Machine Learning classification models on a single public dataset.

Using the **UCI Dry Bean Dataset**, we train **six classification models** and compare their performance using the following evaluation metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

The final goal is to identify which model performs best for this multi-class classification problem and provide observations for each model.

---

## b. Dataset description  

**Dataset Name:** Dry Bean Dataset
**Source:** UCI Machine Learning Repository
**Type:** Multi-class classification

**Dataset Size:**

* Total instances: **13,611** (raw)
* After duplicate removal: **13,543**

**Features:**

* Total features: **16** numeric features
* Target: **Class** (7 bean types)

**Target Classes (7):**

* BARBUNYA
* BOMBAY
* CALI
* DERMASON
* HOROZ
* SEKER
* SIRA

**Preprocessing performed:**

* Removed duplicate rows
* Checked missing values (none after cleaning)
* Label Encoding applied on the target column (`Class`)
* Standard Scaling applied only for models that require it (Logistic Regression, kNN, SVM)

---

## c. Models used 

The following 6 models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Naive Bayes Classifier (GaussianNB)
5. Random Forest Classifier (Ensemble)
6. XGBoost Classifier (Ensemble)

---

### Comparison Table (Evaluation Metrics)

| ML Model Name            | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression      |   0.9192 | 0.9947 |    0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree            |   0.8966 | 0.9465 |    0.8965 | 0.8966 | 0.8964 | 0.8750 |
| kNN                      |   0.9155 | 0.9832 |    0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes              |   0.8970 | 0.9911 |    0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Random Forest (Ensemble) |   0.9192 | 0.9931 |    0.9192 | 0.9192 | 0.9191 | 0.9022 |
| XGBoost (Ensemble)       |   0.9199 | 0.9949 |    0.9202 | 0.9199 | 0.9200 | 0.9031 |

---

### Observations on model performance 

| ML Model Name            | Observation about model performance                                                                                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression performed strongly after feature scaling. It achieved high Accuracy and one of the best MCC scores, showing it can separate the Dry Bean classes effectively.                  |
| Decision Tree            | Decision Tree gave the weakest overall performance. It is likely overfitting some patterns while failing to generalize well, which reduced Accuracy and MCC compared to other models.              |
| kNN                      | kNN performed well and produced results close to Logistic Regression. Since this dataset has well-separated numeric features, distance-based classification works effectively after scaling.       |
| Naive Bayes              | Naive Bayes achieved moderate performance. Its assumptions (feature independence and Gaussian distribution) do not perfectly match the dataset, leading to lower Accuracy and MCC than top models. |
| Random Forest (Ensemble) | Random Forest performed strongly and matched Logistic Regression closely. As an ensemble of trees, it improved generalization compared to a single Decision Tree and achieved high MCC.            |
| XGBoost (Ensemble)       | XGBoost achieved the best overall performance among all models. Its boosting approach improves weak learners iteratively, resulting in the highest Accuracy and MCC on this dataset.               |

---

## How to Run the Project

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run Streamlit app

```bash
python -m streamlit run streamlit_app.py
```

---

## Project Structure

```text
2025AA05530_ML_ASSIGNMENT_2/
│
├── model/
│   ├── data_loader.py
│   ├── data_health_check.py
│   ├── preprocessing.py
│   └── trainer.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Notes

* The dataset is loaded directly from the UCI repository via URL.
* The dataset is multi-class, so AUC is calculated using **One-vs-Rest (OvR)** strategy with macro averaging.
* Scaling is applied only for distance/gradient based models (Logistic Regression and kNN).
