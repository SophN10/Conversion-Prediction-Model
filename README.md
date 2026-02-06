# User Conversion Prediction

This project analyses anonymised e-commerce user behaviour and site interaction data to predict whether a website visitor will convert (make a purchase).  
The objective is to demonstrate how applied machine learning can support data-driven decision-making in marketing and digital product optimisation.

---

## Problem Statement
Understanding which user behaviours lead to conversion allows organisations to:
- prioritise high-intent users  
- improve targeting and personalisation strategies  
- optimise website design and user journeys  

This project frames conversion prediction as a **binary classification problem** using behavioural and interaction-level features.

---

## Data Overview
The dataset contains anonymised user-level interaction data, including:
- browsing activity  
- clickstream events  
- engagement and purchase indicators  

The target variable indicates whether a user converted or not.  
Class imbalance is present and explicitly handled during modelling and evaluation.

---

## Approach
The project follows an end-to-end applied machine learning workflow:

### 1. Data Preparation & Exploration
- Data cleaning and preprocessing  
- Feature scaling and transformation  
- Handling class imbalance  
- Exploratory data analysis (EDA)

### 2. Model Development
Multiple models were trained and compared:
- **Logistic Regression** – baseline model for interpretability  
- **Random Forest** – to capture non-linear feature interactions  
- **XGBoost** – high-performance gradient boosting classifier  

### 3. Model Evaluation
Models were evaluated using metrics suitable for imbalanced classification:
- Precision-Recall AUC (PR-AUC)  
- Precision, Recall and F1-score  

Performance trade-offs were assessed to align model choice with business objectives.

---

## Results & Evaluation
The final model demonstrated strong performance in identifying high-intent users while managing false positives.

**Key evaluation metrics (best performing model):**
- PR-AUC: 0.99  
- Precision: 0.94  
- Recall: 0.96  
- F1-score: 0.95 

These results show how predictive models can support **targeted marketing and conversion optimisation strategies**.

---

## Model Explainability & Insights
Model explainability techniques were applied to ensure predictions could be interpreted and communicated effectively.

- Feature importance and SHAP values were used to identify the strongest drivers of conversion  
- Behavioural signals such as engagement intensity, browsing patterns, and purchase-related interactions were among the most influential features  

These insights help bridge model outputs with actionable business understanding and support stakeholder trust in the model.

Visualisations of feature importance and SHAP results are included in the accompanying presentation.

---

## Project Structure
- `01_data_cleaning.R` – Data cleaning, feature engineering, class balancing and EDA  
- `02_model_training.R` – Model training, evaluation and comparison  
- `presentation/` – PDF presentation summarising methodology, results and explainability insights  

---

## How to Run
1. Clone the repository  
2. Open the `.R` scripts in RStudio (or another R IDE)  
3. Install required packages (e.g. tidyverse, caret, xgboost)  
4. Run scripts in order:  
   `01_data_cleaning.R` → `02_model_training.R`

---

## Tools & Technologies
- R  
- tidyverse  
- caret  
- xgboost  
- Machine learning and model evaluation techniques  

---

## Limitations & Next Steps
- Incorporate additional session-level or temporal behavioural features  
- Operationalise the model by exposing predictions via a simple API or interactive dashboard for ongoing decision support  

---

## Goal
To demonstrate an end-to-end applied machine learning project that connects **user behaviour, predictive modelling, explainability, and business decision-making**.

Project presentation (PDF):  
[View presentation](Online_Shopping_Conversion_Project.pdf)

