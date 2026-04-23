Real-Time Customer Purchase Intent Prediction

Predicting customer purchase intent using behavioral session data. This project goes beyond standard classification by engineering domain-specific engagement features and applying cost-sensitive learning to address class imbalance (~15% conversion rate).

# Key Results:
ROC-AUC: 91.7%
Improved recall on the minority (conversion) class via class-weight tuning and threshold optimization
Deployed an interactive dashboard for real-time decision support using Streamlit

# Problem Overview

In e-commerce, only a small fraction of sessions lead to purchases. This imbalance makes standard classification models biased toward predicting “no purchase.”

This project focuses on:
Identifying high-intent users in real time
Minimizing missed conversions (false negatives)
Providing actionable insights for marketing teams

# Solution Architecture

## Feature Engineering

Custom behavioral features were designed to capture user intent:

Product_Efficiency = time spent on product pages / number of product pages
Admin_Ratio = administrative actions / total session actions (proxy for friction)
PageValues = estimated revenue contribution of visited pages

These features encode engagement quality, not just raw activity.

## ML Pipeline
Built using scikit-learn Pipeline for reproducibility and clean deployment:

Preprocessing (scaling, transformations)
Custom feature engineering
Cost-sensitive Random Forest model

Serialization with Joblib

## Cost-Sensitive Learning

To handle class imbalance:

Applied class weights in Random Forest
Tuned decision threshold to prioritize recall for high-value users
This ensures the model captures more potential buyers rather than defaulting to majority predictions.

## Interactive Dashboard

Developed with Streamlit:

Real-time predictions from user session inputs
Adjustable classification threshold
Enables business users to balance precision vs recall dynamically

# Model Evaluation

Validation Strategy: Stratified train-test split + cross-validation

Metrics:

ROC-AUC: 0.917 | Excellent ability to rank high-intent users above low-intent users. 
Recall (Revenue): 0.84 | Captures 84% of potential buyers—crucial for targeted retargeting.
F1-score: 0.81 | Robust balance between precision and recall for stable deployment. 

Focus was placed on recall and business impact, not just overall accuracy.

## Key Insights (Feature Importance)

Top drivers of purchase intent:
PageValues — strongest indicator of conversion likelihood
Product_Efficiency — high engagement per product correlates with intent
Admin_Ratio — elevated values signal friction during checkout

# Tech Stack

Language: Python
Libraries:scikit-learn
Pandas
Joblib
Streamlit
Seaborn
Engineering Practices:
Modular src/ architecture
Object-oriented pipeline design
Model serialization
Version control with Git/GitHub

# How to Run

Clone the repository
    git clone https://github.com/ersiisufi/ShoppersBehaviour

Install dependencies
    pip install -r requirements.txt

Train the model
    python main.py

Launch the dashboard
    streamlit run app.py

# Final Note:

This project emphasizes not just predictive performance, but deployability, interpretability, and business alignment—key aspects of real-world machine learning systems.