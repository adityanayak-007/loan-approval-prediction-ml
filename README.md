# Loan Approval Prediction

A beginner machine learning project that predicts whether a loan application will be **Approved** or **Rejected**, based on an applicant's personal and financial details. Built as a first-year B.Tech (CSE - AI/ML) project.

---

## Problem Statement

Banks and lending platforms need a quick way to estimate whether a loan application is likely to be approved, based on factors like income, credit score, employment history, and loan amount. This project builds a classification model that learns from past loan data and predicts the outcome for a new applicant.

---

## Dataset

- **File**: `loan_data.csv`
- **Source**: [add your dataset link here]
- **Size**: 45,000 applicant records, 14 columns

**Target column**
| Column | Meaning |
|---|---|
| `loan_status` | 0 = Rejected, 1 = Approved (~78% rejected, ~22% approved — an imbalanced dataset) |

**Numerical features**
| Column | Meaning |
|---|---|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_emp_exp` | Years of work experience |
| `loan_amnt` | Loan amount requested |
| `loan_int_rate` | Interest rate on the loan |
| `loan_percent_income` | Loan amount as a fraction of income |
| `cb_person_cred_hist_length` | Length of credit history (years) |
| `credit_score` | Credit bureau score (390–850) |

**Categorical features**
| Column | Meaning |
|---|---|
| `person_gender` | female / male |
| `person_education` | High School / Associate / Bachelor / Master / Doctorate |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `loan_intent` | PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION |
| `previous_loan_defaults_on_file` | No / Yes |

---

## My Approach

**1. Understanding the data**
I started by loading the dataset with pandas and checking its shape, column types, and summary statistics. I also checked the balance of the target column and found it was imbalanced — only about 22% of applications were approved. This mattered later when choosing how to train the model.

**2. Cleaning the data**
There were no missing values, but I found a data quality issue: a few rows had unrealistic ages (like 144 years), which were clearly data entry errors. I removed those rows before training.

**3. Preparing the data for the model**
Machine learning models only understand numbers, not text, so I converted the categorical columns:
- Binary columns (`person_gender`, `previous_loan_defaults_on_file`) were converted to 0/1 using label encoding.
- Multi-category columns (`person_education`, `person_home_ownership`, `loan_intent`) were converted using one-hot encoding (`pd.get_dummies`), since these categories have no natural order.

**4. Exploring the data**
I used bar charts, histograms, and a correlation heatmap to look for patterns before modeling — for example, comparing approval rates across different home ownership types and loan purposes. This step gave me an early sense of which features might matter, though the actual model later revealed some patterns weren't as intuitive as they first looked (more on this in the Results section).

**5. Splitting the data**
I split the data into 80% training and 20% testing using `train_test_split`, with `stratify=y` so both sets kept the same approval/rejection ratio as the full dataset. This gives a fair evaluation on data the model has never seen.

**6. Training the model**
I trained a **Random Forest Classifier** — a model that builds many decision trees and combines their votes for a final prediction. I used `class_weight='balanced'` so the model wouldn't just learn to always predict "rejected" (which would still be ~78% accurate but useless), and instead pays proper attention to the smaller group of approved applicants.

**7. Evaluating the model**
I tested the trained model on the unseen 20% test set and measured accuracy, precision, recall, and F1-score, along with a confusion matrix to see exactly what kinds of mistakes it made. Full numbers are in the [Results](#results-and-model-evaluation) section below.

**8. Building a prediction interface**
I saved the trained model using `joblib` and built a simple Streamlit web app (`app.py`) so anyone can enter an applicant's details into a form and instantly get a prediction, without needing to run any code.

---

## Results and Model Evaluation

Evaluated on the held-out test set (8,999 unseen applicants):

| Metric | Score | What it means |
|---|---|---|
| **Accuracy** | 92.6% | The model's overall guesses were correct 92.6% of the time |
| **Precision** | 90.1% | When the model predicted "Approved," it was right 90.1% of the time |
| **Recall** | 75.2% | Of all applicants who were actually approved, the model correctly identified 75.2% of them |
| **F1-Score** | 81.9% | A balanced measure combining precision and recall |

**Confusion Matrix**

_(Insert the confusion matrix heatmap image from the notebook here — screenshot it and save as e.g. `screenshots/confusion_matrix.png`, then embed with:_
`![Confusion Matrix](screenshots/confusion_matrix.png)`_)_

A confusion matrix shows four things: how many approvals were correctly predicted, how many rejections were correctly predicted, and how many of each were mixed up. This is more informative than accuracy alone, especially since our dataset is imbalanced.

**Most Important Features**

The model relies most heavily on these features when making a decision:
1. `loan_percent_income` — how large the loan is relative to income
2. `loan_int_rate` — the interest rate
3. `person_income` — annual income
4. `credit_score` — credit score

Gender and education had almost no influence on the outcome.

**An interesting/unexpected finding**

While exploring the data, I initially assumed things like higher income, owning a home, or a longer credit history would make approval more likely — that's the common-sense assumption about loans. But testing individual sample profiles through the trained model showed the opposite in several cases: applicants with lower income and higher loan-to-income ratios were sometimes approved more often than "safer-looking" high-income applicants. This showed me that Random Forest doesn't rely on any single rule — it learns from the *combination* of all features together, which can produce results that don't match simple human intuition.

**Example predictions from the app**

| Applicant profile | Prediction | Confidence |
|---|---|---|
| Age 35, Income $85,000, Master's, no prior default, RENT, PERSONAL loan, 15% loan-to-income | Rejected | 77% |
| Age 29, Income $54,464, Associate, no prior default, RENT, VENTURE loan, 26% loan-to-income | Approved | 99% |

_(Add your own screenshots of the running app here for both an approved and a rejected case — this is strong evidence that you tested the model, not just built it.)_
`![Approved Example](screenshots/approved_example.png)`
`![Rejected Example](screenshots/rejected_example.png)`

---

## Project Structure

```
loan_prediction_beginner/
├── loan_data.csv          # Dataset file
├── loan_prediction.ipynb  # Jupyter notebook (all code, charts, explanations)
├── model.joblib           # Saved trained Random Forest model
├── app.py                 # Streamlit web app for predictions
├── requirements.txt       # Required Python libraries
├── screenshots/           # App screenshots used in this README
└── README.md              # This file
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the notebook** (to see all code, charts, and explanations)
```bash
jupyter notebook loan_prediction.ipynb
```

**3. Run the web app**
```bash
python -m streamlit run app.py
```
> Note: if `streamlit run app.py` gives a "command not recognized" error (common on Windows/PowerShell when Streamlit isn't on your system PATH), use the command above instead — it runs Streamlit as a Python module and works reliably.

Fill in the applicant details, click **Predict**, and it will show whether the loan is Approved or Rejected along with a confidence score.

---

## Possible Future Improvements

- Try other models (Logistic Regression, Decision Tree, SVM) and compare performance
- Tune hyperparameters with GridSearchCV for potentially better recall
- Add SHAP or LIME explanations so the app can show *why* a specific prediction was made
