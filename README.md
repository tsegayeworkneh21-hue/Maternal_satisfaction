# Maternal Satisfaction Prediction

## Overview

This project applies **Machine Learning (Logistic Regression)** to predict maternal satisfaction with delivery-care services in **Bahir Dar city health facilities, Northwest Ethiopia**.

The project uses the maternal delivery-care satisfaction dataset and develops a reduced **10-feature Logistic Regression model** that can be deployed as an interactive **Streamlit** application.

The project workflow is:

```text
Original dataset
      ↓
Data understanding and management
      ↓
27 candidate predictors
      ↓
Feature selection
      ↓
10 selected predictors
      ↓
Logistic Regression
      ↓
Hyperparameter tuning
      ↓
Model evaluation
      ↓
Threshold analysis
      ↓
Streamlit deployment
```

---

## Project Objectives

The main objectives are to:

1. Analyze maternal satisfaction with delivery-care services.
2. Perform data management and prepare variables for machine learning.
3. Develop a Logistic Regression model for predicting maternal satisfaction.
4. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - ROC-AUC
   - F1 Score
   - Brier Score
5. Identify important predictors using permutation importance.
6. Deploy the final model using Streamlit.

---

## Dataset

The project uses a real maternal delivery-care satisfaction dataset in SPSS format.

### Dataset characteristics

- **Observations:** 894
- **Original variables:** 46
- **Candidate predictors:** 27
- **Final selected predictors:** 10
- **Target variable:** `satisfaction`

### Target coding

The original satisfaction variable was converted to a binary target:

| Value | Meaning | Machine-learning class |
|---|---|---:|
| 1 | Not satisfied | 0 |
| 2 | Satisfied | 1 |

The dataset contains:

- **345 (38.6%)** not satisfied
- **549 (61.4%)** satisfied

---

## Candidate Predictors

After data management, 27 candidate predictors were retained:

```text
q101age
q102mstatu
q103religi
q104ethini
q105educat
q106occupa
q107mincom
q108reside
q1091distk
q110reason
q111timeta
q112waitpl
q113privac
q201parity
q202cpplan
q203ancatt
q204hivres
q205ga
q207nstatu
q208pdhsta
q209nsex
q2010bw
q301greeti
q302respec
q303delive
q411owners
modeofdeliveryrecode
```

The candidate predictors include demographic, socioeconomic, obstetric, delivery-care, and service-related variables.

---

## Feature Selection

Two feature-selection approaches were investigated:

1. **SelectKBest**
2. **Permutation Importance**

The reduced model based on permutation importance produced the stronger overall results in the current analysis and was therefore retained as the final reduced model.

### Final 10 selected features

| # | Variable | Description |
|---:|---|---|
| 1 | `q411owners` | Facility ownership |
| 2 | `q112waitpl` | Availability of waiting area |
| 3 | `q301greeti` | Professional greeting |
| 4 | `q113privac` | Privacy during physical examination |
| 5 | `q111timeta` | Time taken to get a professional's attention |
| 6 | `q110reason` | Reason for visiting the hospital |
| 7 | `q108reside` | Residence |
| 8 | `q302respec` | Respectful treatment during delivery |
| 9 | `q1091distk` | Distance to the facility in kilometres |
| 10 | `q102mstatu` | Marital status |

### Why permutation importance?

Permutation importance measures how model performance changes when a feature is randomly shuffled.

A larger decrease in ROC-AUC after shuffling indicates that the feature contributes more to the model's predictive performance.

The final ranking was:

| Rank | Feature | Mean importance |
|---:|---|---:|
| 1 | `q112waitpl` | 0.060277 |
| 2 | `q113privac` | 0.046489 |
| 3 | `q301greeti` | 0.038465 |
| 4 | `q411owners` | 0.034218 |
| 5 | `q1091distk` | 0.020077 |
| 6 | `q111timeta` | 0.010059 |
| 7 | `q110reason` | 0.009840 |
| 8 | `q302respec` | 0.009163 |
| 9 | `q102mstatu` | 0.004370 |
| 10 | `q108reside` | 0.002238 |

**Important:** permutation importance indicates predictive contribution. It does **not** prove that a feature has a causal effect on maternal satisfaction.

---

## Machine Learning Model

The final model is:

**Logistic Regression**

The model uses a preprocessing pipeline:

### Numerical variable

`q1091distk`

- Standardized using `StandardScaler`.

### Categorical variables

The remaining nine selected variables are categorical.

They are transformed using:

```python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

The preprocessing and Logistic Regression classifier are stored together in a single scikit-learn pipeline.

---

## Train/Test Split

The dataset was divided using an 80/20 stratified split:

| Dataset | Observations |
|---|---:|
| Training | 715 |
| Testing | 179 |
| Total | 894 |

Stratification was used so that the distribution of satisfied and not-satisfied mothers was maintained between training and testing data.

---

## Hyperparameter Tuning

The Logistic Regression regularization parameter `C` was tuned using:

- **5-fold StratifiedKFold cross-validation**
- **GridSearchCV**
- Scoring metric: **precision**

The values evaluated were:

```python
[0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
```

The best value found was:

```text
C = 2.0
```

The model also used:

```python
class_weight="balanced"
max_iter=1000
```

---

## Model Performance

The final reduced Logistic Regression model achieved the following results on the current test evaluation:

| Metric | Result |
|---|---:|
| Accuracy | **85.47%** |
| Precision | **88.18%** |
| Recall | **88.18%** |
| ROC-AUC | **0.9165** |
| F1 Score | **0.8818** |
| Brier Score | **0.1163** |

### Interpretation

- **Accuracy = 85.47%:** 85.47% of test observations were classified correctly.
- **Precision = 88.18%:** among observations predicted as satisfied, 88.18% were actually satisfied.
- **Recall = 88.18%:** 88.18% of truly satisfied mothers were correctly identified.
- **ROC-AUC = 0.9165:** the model has strong ability to discriminate between the two classes.
- **F1 = 0.8818:** indicates a good balance between precision and recall.
- **Brier = 0.1163:** measures the quality of predicted probabilities; lower values are better.

---

## Confusion Matrix

At the default threshold of 0.50:

```text
                 Predicted
                 Not Sat.   Satisfied

Actual Not Sat.      56          13
Actual Satisfied     13          97
```

Therefore:

- True Negative (TN) = 56
- False Positive (FP) = 13
- False Negative (FN) = 13
- True Positive (TP) = 97

The recall for the **not-satisfied class** is:

```text
56 / (56 + 13) = 81.16%
```

The recall for the **satisfied class** is:

```text
97 / (97 + 13) = 88.18%
```

---

## Threshold Analysis

The Streamlit application allows the user to change the decision threshold.

The default threshold is:

```text
0.50
```

The notebook tested thresholds from 0.20 to 0.60.

| Threshold | Not Satisfied Recall | Satisfied Recall | Accuracy | F1 |
|---:|---:|---:|---:|---:|
| 0.20 | 47.83% | 99.09% | 79.33% | 85.49% |
| 0.25 | 49.28% | 99.09% | 79.89% | 85.83% |
| 0.30 | 50.72% | 99.09% | 80.45% | 86.17% |
| 0.35 | 56.52% | 96.36% | 81.01% | 86.18% |
| 0.40 | 65.22% | 93.64% | 82.68% | 86.92% |
| 0.45 | 69.57% | 90.91% | 82.68% | 86.58% |
| **0.50** | **81.16%** | **88.18%** | **85.47%** | **88.18%** |
| 0.55 | 82.61% | 85.45% | 84.36% | 87.04% |
| 0.60 | 84.06% | 81.82% | 82.68% | 85.31% |

For the current project, **0.50** is used as the default because it provides a good overall balance.

If the main objective is to identify more mothers who are potentially not satisfied, a higher threshold can be considered. However, changing the threshold creates a trade-off between the two classes.

---

## Streamlit Application

The project includes a Streamlit deployment application.

The application:

1. Loads the trained model.
2. Provides user-friendly input fields.
3. Converts the displayed choices into the exact numerical SPSS codes expected by the model.
4. Sends the inputs to the model in the correct feature order.
5. Predicts:
   - Satisfied
   - Not satisfied
6. Displays the predicted probability of satisfaction.
7. Allows the user to change the decision threshold.

The application uses **Python and Streamlit only**.

---

## Project Structure

Place the project files in the same directory:

```text
Maternal-Satisfaction-Project/
│
├── MSMLP(1).ipynb
├── capp.py
├── Maternal_Satisfaction_Logistic_Model_10Features.pkl
├── requirements.txt
└── README.md
```

### File descriptions

| File | Purpose |
|---|---|
| `MSMLP(1).ipynb` | Complete analysis and machine-learning notebook |
| `capp.py` | Streamlit deployment application |
| `Maternal_Satisfaction_Logistic_Model_10Features.pkl` | Trained Logistic Regression pipeline |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |

---

## Installation

### 1. Clone or download the project

Open a terminal and move into the project directory:

```bash
cd Maternal-Satisfaction-Project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib streamlit pyreadstat matplotlib plotly
```

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Application

Make sure the following two files are in the same folder:

```text
capp.py
Maternal_Satisfaction_Logistic_Model_10Features.pkl
```

Then run:

```bash
streamlit run capp.py
```

Streamlit will start the application and provide a local address, normally similar to:

```text
http://localhost:8501
```

Open the address in your browser.

---

## Using the Application

### Step 1

Run:

```bash
streamlit run capp.py
```

### Step 2

Enter the mother's information using the input fields.

The application uses these ten features:

1. Facility ownership
2. Waiting area
3. Professional greeting
4. Privacy
5. Time to professional attention
6. Reason for hospital visit
7. Residence
8. Respectful treatment
9. Distance to facility
10. Marital status

### Step 3

Choose the decision threshold if needed.

The default is:

```text
0.50
```

### Step 4

Click:

```text
Predict satisfaction
```

The application displays:

- Predicted class
- Predicted probability of satisfaction
- Exact values sent to the model

---

## Important Model Compatibility Requirement

The model was trained using the **raw numerical SPSS codes**, not decoded text labels.

For example:

```text
Facility ownership:
Private    → 1.0
Government → 2.0
```

and:

```text
Residence:
Urban  → 1.0
Rural  → 2.0
```

The Streamlit application handles these conversions automatically.

Do **not** manually replace the model inputs with arbitrary text or different numerical codes.

The input feature order is:

```python
[
    "q411owners",
    "q112waitpl",
    "q301greeti",
    "q113privac",
    "q111timeta",
    "q110reason",
    "q108reside",
    "q302respec",
    "q1091distk",
    "q102mstatu"
]
```

---

## Reproducing the Model

The notebook contains the analysis workflow, including:

```text
Data loading
     ↓
Data inspection
     ↓
Descriptive analysis
     ↓
Data management
     ↓
Target encoding
     ↓
Candidate feature selection
     ↓
Train/test split
     ↓
Preprocessing
     ↓
Logistic Regression
     ↓
Feature selection
     ↓
Reduced model
     ↓
GridSearchCV
     ↓
Model evaluation
     ↓
Threshold analysis
     ↓
Permutation importance
     ↓
Model saving
```

The trained model is saved with:

```python
joblib.dump(
    logistic_model_reduced,
    "Maternal_Satisfaction_Logistic_Model_10Features.pkl"
)
```

---

## Model Limitations

This project is primarily an academic machine-learning project and should not be interpreted as a validated clinical decision-support system.

Important limitations include:

### 1. Test-set feature selection

The current notebook calculates permutation importance using the test set and then uses the selected features in the reduced model. Because the same test set is subsequently used for performance evaluation, this can introduce **selection bias**.

For a publication-grade analysis, feature selection should be performed using training data or within cross-validation, while the final test set remains completely untouched.

### 2. Cross-sectional data

The underlying dataset is cross-sectional. Therefore, model importance should not be interpreted as causal effects.

### 3. External validation

The model should be tested on an independent dataset from other health facilities or a later time period before real-world implementation.

### 4. Threshold selection

Threshold analysis was performed on the available test predictions. A more rigorous workflow would select the operating threshold using validation data or cross-validation and then evaluate the fixed threshold on the untouched test set.

### 5. Missing data

Some variables in the original dataset had substantial missingness. For example, distance in hours had 64.21% missing observations, while the walking-distance-hours recode had 69.24% missingness and limited variation. These variables were not used in the initial predictive model.

---

## Ethical and Practical Considerations

The model should be used as a **decision-support or research tool**, not as a replacement for clinical judgment or direct communication with mothers.

A prediction of "satisfied" does not mean that a mother is definitely satisfied, and a prediction of "not satisfied" does not establish the reason for dissatisfaction.

Health facilities should continue to assess women's experiences directly.

---

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- pyreadstat
- joblib
- Matplotlib
- Plotly
- Streamlit

---

## Main Python Libraries

```python
import pandas as pd
import numpy as np
import pyreadstat
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance
```

---

## Project Outcome

The final project produces:

```text
                    Maternal Satisfaction
                             │
                             ↓
                  Logistic Regression
                             │
             ┌───────────────┴───────────────┐
             ↓                               ↓
       Not Satisfied                     Satisfied
             │                               │
             └───────────────┬───────────────┘
                             ↓
                   Predicted probability
                             ↓
                     Streamlit interface
```

The current model provides a compact 10-input prediction interface while retaining strong predictive performance in the current evaluation.

---

## Author / Academic Project

**Project:** Maternal Satisfaction among Vaginal and Cesarean Section Delivery Care Services in Bahir Dar City Health Facilities, Northwest Ethiopia: Application of Machine Learning

**Model:** Logistic Regression

**Deployment:** Streamlit

**Target:** Maternal satisfaction

**Dataset size:** 894 observations

**Final predictors:** 10

---

## Disclaimer

This application is developed for **academic and research purposes**. It has not been clinically validated and should not be used as an independent basis for medical or healthcare decisions.

For real-world deployment, the model should undergo appropriate external validation, bias assessment, data-protection review, clinical evaluation, and governance approval.
