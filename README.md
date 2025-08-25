# User Conversion Predictions 
This project analyses anonymised user behaviour and site interaction data from an ecommerce context to predict whether a visitor will covert or not. 
Features include browsing activity, clickstream events and purchase indicators. 

# Project Structure 
01_data_cleaning.R** - Cleans raw data, class balancing and scaling and performs explorator data analysis (EDA).
02_model_training.R** - Trained logstic regression model (baseline for interpretability), random forest (capture non-linear interactions), 
xgboost (high-performance classification) and evaluated using PR-AUC (handle imbalance) and precision, recall, F1 score (assess classification quality). 

# How to run 
1. Clone the repository or download the files.
2. Open each `.R` file in RStudio or another R IDE.
3. Install required R packages (tidyverse, caret, etc.).
4. Run scripts in order: 01 -> 02

# Tools and Packages 
- R, tidyverse, caret (edit to match your packages)
- Data cleaning, feature engineering, modeling, evaluation

# Goal 
Understand which user and site behaviours lead to conversion and build a predictive model to support marketing and site functionality decisions.
