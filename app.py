from src.feature_engineering import engineer_behavioral_features
import streamlit as st
import pandas as pd
import joblib
import config

# 1. Page Setup
st.set_page_config(page_title="Purchase Intent AI", page_icon="💰")

# 2. Load the trained Pipeline
@st.cache_resource
def load_pipeline():
    # Make sure this path matches where main.py saved your model!
    return joblib.load("models/model.joblib")

model = load_pipeline()

# 3. Header
st.title("🛍️ E-Commerce Purchase Intent Predictor")
st.markdown("""
This tool uses a **Weighted Random Forest** to predict if a website session will end in a purchase.
Adjust the **Top 5 Drivers** below to see how the probability changes.
""")

# 4. Input Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("High-Impact Metrics")
    threshold = st.slider("Targeting Sensitivity (Threshold)", 0.1, 0.9, 0.3)
    # These are our "Top Drivers" from the Feature Importance analysis
    page_values = st.slider("Page Values (Avg. value of pages visited)", 0.0, 300.0, 20.0)
    exit_rates = st.slider("Exit Rate (Percentage of exits from these pages)", 0.0, 0.2, 0.02)
    product_related = st.number_input("Product Related Pages Visited", 0, 500, 15)
    prod_duration = st.number_input("Total Time on Product Pages (Seconds)", 0, 10000, 300)
    admin_pages = st.slider("Administrative Pages Visited", 0, 20, 2)

with col2:
    st.subheader("Prediction Result")
    
    # --- Create the Full Feature Row ---
    input_data = {
        'Administrative': admin_pages,
        'Administrative_Duration': 40.0,
        'Informational': 0,
        'Informational_Duration': 0.0,
        'ProductRelated': product_related,
        'ProductRelated_Duration': prod_duration,
        'BounceRates': 0.01,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': 0.0,
        'Month': 'May',
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 1,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }
    
    input_df = pd.DataFrame([input_data])
    

    prob = model.predict_proba(input_df)[0][1]
    is_purchase = prob >= threshold
    # Display the Result
    st.metric(label="Purchase Probability", value=f"{prob:.1%}")
    

    if is_purchase:
        st.success(f"🎯 TARGET: High potential at {prob:.1%}")
    else:
        st.info(f"⏳ NURTURE: Low potential at {prob:.1%}")

# 5. The "Why" behind the prediction
st.divider()
st.info(f"**Pro Insight:** The model identifies {'PageValues'} as the strongest driver. Currently, this session has a PageValue of {page_values}.")