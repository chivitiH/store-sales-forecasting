"""
Store Sales Forecasting - Dashboard Streamlit
Interface interactive pour visualiser les prédictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Store Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

# Paths
MODELS_PATH = Path("models")
DATA_PATH = Path("data")

# Title
st.title("🛒 Store Sales Time Series Forecasting")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Load data
@st.cache_data
def load_data():
    train = pd.read_csv(DATA_PATH / "processed/train_processed.csv", parse_dates=['date'])
    return train

@st.cache_resource
def load_model(model_name):
    if model_name == "XGBoost":
        return joblib.load(MODELS_PATH / "xgboost_full_model.pkl")
    elif model_name == "LightGBM":
        return joblib.load(MODELS_PATH / "lightgbm_full_model.pkl")
    return None

# Load results
@st.cache_data
def load_results():
    return pd.read_csv(MODELS_PATH / "all_models_results.csv")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🤖 Models", "📈 Predictions"])

with tab1:
    st.header("📊 Dataset Overview")
    
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Stores", df['store_nbr'].nunique())
    with col3:
        st.metric("Product Families", df['family'].nunique())
    with col4:
        st.metric("Total Sales", f"${df['sales'].sum():,.0f}")
    
    st.markdown("---")
    
    # Sales over time
    st.subheader("Sales Over Time")
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    fig = px.line(daily_sales, x='date', y='sales', 
                  title='Daily Total Sales (2013-2017)')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top families
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Product Families")
        top_families = df.groupby('family')['sales'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_families.values, y=top_families.index, orientation='h',
                     labels={'x': 'Total Sales', 'y': 'Product Family'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Store Type")
        store_sales = df.groupby('type')['sales'].sum().sort_values(ascending=False)
        fig = px.pie(values=store_sales.values, names=store_sales.index)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🤖 Model Performance")
    
    results = load_results()
    
    # Metrics
    st.subheader("Model Comparison")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=results['model'],
        y=results['rmse'],
        name='RMSE',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title='Model Comparison - RMSE',
        xaxis_title='Model',
        yaxis_title='RMSE',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("Detailed Results")
    st.dataframe(
        results.style.highlight_min(subset=['rmse', 'mae', 'mape'], color='lightgreen'),
        use_container_width=True
    )
    
    # Best model
    best_model = results.iloc[0]
    st.success(f"🏆 Best Model: **{best_model['model']}** with MAPE = **{best_model['mape']:.2f}%**")

with tab3:
    st.header("📈 Make Predictions")
    
    st.info("⚠️ This is a demo interface. For production use, implement API endpoints.")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["XGBoost", "LightGBM"]
    )
    
    # Load selected model
    model = load_model(model_choice)
    
    if model:
        st.success(f"✅ {model_choice} model loaded")
        
        # Sample prediction
        st.subheader("Sample Prediction")
        
        df = load_data()
        sample = df.sample(1)
        
        st.write("Input Features:")
        st.json(sample[['family', 'store_nbr', 'date', 'onpromotion']].to_dict('records')[0])
        
        if st.button("Predict"):
            feature_cols = [
                'dayofweek', 'day', 'month', 'quarter', 'dayofyear', 'weekofyear',
                'is_weekend', 'is_month_start', 'is_month_end', 'is_payday',
                'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
                'sales_rolling_mean_7', 'sales_rolling_std_7',
                'sales_rolling_mean_14', 'sales_rolling_std_14',
                'sales_rolling_mean_30', 'sales_rolling_std_30',
                'is_holiday', 'has_promotion', 'onpromotion',
                'cluster', 'transactions', 'type_encoded'
            ]
            
            feature_cols = [col for col in feature_cols if col in sample.columns]
            X = sample[feature_cols].fillna(0)
            
            prediction = model.predict(X)[0]
            actual = sample['sales'].values[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Sales", f"${prediction:.2f}")
            with col2:
                st.metric("Actual Sales", f"${actual:.2f}")
            with col3:
                error = abs(actual - prediction) / actual * 100
                st.metric("Error", f"{error:.2f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Store Sales Forecasting**

Time Series ML Project

Built with:
- XGBoost
- LightGBM
- TensorFlow
- Streamlit
""")
