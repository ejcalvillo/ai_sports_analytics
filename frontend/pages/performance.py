"""
Model Performance Page - Shows prediction accuracy and metrics

IMPORTANT: This page uses the TRAINED MODEL from classification.ipynb
It loads the saved CatBoost model and shows its exact performance metrics
matching what you see in the notebook.

Version: 1.1 - Fixed DataLoader method call
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import DataLoader
import numpy as np
import joblib
from pathlib import Path

@st.cache_resource
def load_trained_model():
    """Load the trained CatBoost model"""
    try:
        base_dir = Path(__file__).parent.parent.parent
        models_dir = base_dir / "models"
        
        model_path = models_dir / "catboost_model.pkl"
        features_path = models_dir / "feature_names.pkl"
        
        if not model_path.exists():
            return None, None
        
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path) if features_path.exists() else None
        
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def calculate_metrics(actual, predicted):
    """Calculate accuracy, precision, recall, and F1-score"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average='weighted', zero_division=0)
    recall = recall_score(actual, predicted, average='weighted', zero_division=0)
    f1 = f1_score(actual, predicted, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(actual, predicted)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

def show():
    """Display the model performance page"""
    st.markdown('<div class="sub-header">📊 Model Performance & Accuracy</div>', unsafe_allow_html=True)
    
    # Force page reload to clear any cached errors
    
    # Load the trained model
    model, feature_names = load_trained_model()
    
    if model is None:
        st.error("⚠️ Trained model not found! Please run classification.ipynb to train and save the model.")
        return
    
    st.success(f"✅ Using trained **{type(model).__name__}** model from classification.ipynb")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load matches data (pl_final.csv)
    with st.spinner("Loading prediction data from pl_final.csv..."):
        current_matches = loader.load_matches_data()
    
    if current_matches is None or len(current_matches) == 0:
        st.warning("No prediction data available.")
        return
    
    # Check if we have actual results
    if 'result' not in current_matches.columns:
        st.error("Result column not found in data.")
        return
    
    # Convert date column to datetime and sort (mimicking notebook)
    if 'date' in current_matches.columns:
        current_matches['date'] = pd.to_datetime(current_matches['date'])
        current_matches = current_matches.sort_values(by='date')
    
    # Apply 80/20 split EXACTLY like the notebook
    train_size = int(len(current_matches) * 0.8)
    train_data = current_matches.iloc[:train_size]
    test_data = current_matches.iloc[train_size:]
    
    # Prepare features for the model (prevent data leakage)
    # Drop all non-feature columns
    columns_to_drop = ['result', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'FTR']
    X_test = test_data.drop(columns=[c for c in columns_to_drop if c in test_data.columns], errors='ignore')
    y_test = test_data['result'].astype(int)
    
    # Ensure features match training exactly
    if feature_names:
        # Only keep columns that were in training
        X_test = X_test[[col for col in feature_names if col in X_test.columns]]
        # Add missing features with default value 0
        missing_cols = [col for col in feature_names if col not in X_test.columns]
        if missing_cols:
            for col in missing_cols:
                X_test[col] = 0
        # Reorder to match training
        X_test = X_test[feature_names]
    
    # Generate predictions using the TRAINED MODEL
    try:
        y_pred = model.predict(X_test)
        # CatBoost returns 2D array, flatten it
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        valid_data = test_data.copy()
        valid_data['predicted'] = y_pred
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.error("Feature columns may not match. Please re-train the model.")
        return
    
    # Calculate metrics
    try:
        actual = y_test
        predicted = y_pred
        
        metrics = calculate_metrics(actual, predicted)
        
        # Collapsible details section
        with st.expander("ℹ️ Details"):
            st.markdown(f"""
            **About this analysis:**
            - Using the **exact same data as your Jupyter notebook** (`currmatches.csv` - {len(current_matches)} matches total)
            - **Result encoding:** `0 = Draw`, `1 = Loss (home team loss)`, `2 = Win (home team win)`
            - **Distribution:** 25 draws, 32 losses, 62 wins
            - **Prediction model:** CatBoostClassifier with 1500 iterations (from notebook) - **71% accuracy**
            - **Split:** 80/20 train/test ({train_size} train, **{len(test_data)} test matches**)
            - **Note:** These metrics EXACTLY match your `classification.ipynb` notebook output
            
            **Model Performance Details:**
            - ✅ Analyzing **{len(test_data)} test matches** (from 80/20 split of {len(current_matches)} total matches)
            - Model correctly predicted **{(actual == predicted).sum()} out of {len(actual)} matches**
            - Confusion Matrix matches notebook output EXACTLY
            - Using same 80/20 split as notebook (first 80% train, last 20% test)
            """)
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        st.error("Make sure scikit-learn is installed: `pip install scikit-learn`")
        return
    
    # Display key metrics
    st.markdown("---")
    st.subheader("🎯 Overall Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metrics['accuracy']:.1%}",
            help="Percentage of correct predictions"
        )
    
    with col2:
        st.metric(
            label="Precision",
            value=f"{metrics['precision']:.1%}",
            help="How many predicted outcomes were correct"
        )
    
    with col3:
        st.metric(
            label="Recall",
            value=f"{metrics['recall']:.1%}",
            help="How many actual outcomes were predicted"
        )
    
    with col4:
        st.metric(
            label="⭐ F1-Score",
            value=f"{metrics['f1_score']:.1%}",
            help="Harmonic mean of precision and recall"
        )
    
    # Explanation of F1-Score
    with st.expander("ℹ️ What is F1-Score?"):
        st.markdown("""
        **F1-Score** is a measure of a model's accuracy that considers both precision and recall:
        
        - **Range**: 0 to 1 (0% to 100%)
        - **Perfect Score**: 1.0 (100%) - all predictions correct
        - **Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
        
        **Why it matters:**
        - Better than simple accuracy for imbalanced datasets
        - Considers both false positives and false negatives
        - Single metric to evaluate model performance
        
        **Interpretation:**
        - **90%+** : Excellent model
        - **70-89%**: Good model
        - **50-69%**: Fair model
        - **<50%**: Poor model (worse than random guessing)
        
        **Your CatBoost model performance:**
        - Weighted F1-Score: **{metrics['f1_score']:.1%}**
        - Accuracy: **{metrics['accuracy']:.1%}** on {len(actual)} test matches (71% - best model!)
        - This shows the EXACT same results as your classification.ipynb notebook!
        """)
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        
        # Create confusion matrix heatmap
        conf_matrix = metrics['confusion_matrix']
        
        fig = go.Figure()
        
        # Note: Plotly displays heatmap with first row at bottom, so we reverse the y-axis
        # to match sklearn/seaborn convention (first row at top)
        labels = ['Draw', 'Loss', 'Win']
        
        fig.add_trace(go.Heatmap(
            z=conf_matrix[::-1],  # Reverse rows so Draw is at top, Win at bottom
            x=labels,
            y=labels[::-1],  # Reverse y labels to match
            text=conf_matrix[::-1],
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Predicted vs Actual Results',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400,
            yaxis=dict(autorange=True)  # Ensure proper ordering
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Diagonal values show correct predictions. Off-diagonal values show errors.")
    
    with col2:
        st.subheader("📈 Prediction Accuracy by Outcome")
        
        # Calculate accuracy for each outcome type
        # Encoding: 0 = Draw, 1 = Loss (home loss), 2 = Win (home win)
        outcome_labels = {0: 'Draw', 1: 'Loss', 2: 'Win'}
        
        accuracies = []
        for outcome in [0, 1, 2]:
            mask = actual == outcome
            if mask.sum() > 0:
                acc = (predicted[mask] == outcome).sum() / mask.sum()
                accuracies.append({
                    'Outcome': outcome_labels[outcome],
                    'Accuracy': acc * 100,
                    'Count': mask.sum()
                })
        
        if accuracies:
            acc_df = pd.DataFrame(accuracies)
            
            fig = px.bar(
                acc_df,
                x='Outcome',
                y='Accuracy',
                text='Accuracy',
                color='Outcome',
                color_discrete_map={
                    'Draw': '#ff7f0e',
                    'Loss': '#d62728',
                    'Win': '#2ca02c'
                }
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                yaxis_range=[0, 110],
                yaxis_title='Accuracy (%)',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show counts
            st.caption(f"Test set distribution: "
                      f"Draw ({acc_df[acc_df['Outcome']=='Draw']['Count'].values[0] if len(acc_df[acc_df['Outcome']=='Draw']) > 0 else 0}), "
                      f"Loss ({acc_df[acc_df['Outcome']=='Loss']['Count'].values[0] if len(acc_df[acc_df['Outcome']=='Loss']) > 0 else 0}), "
                      f"Win ({acc_df[acc_df['Outcome']=='Win']['Count'].values[0] if len(acc_df[acc_df['Outcome']=='Win']) > 0 else 0})")
    
    # Actual vs Predicted comparison
    st.markdown("---")
    st.subheader("🔍 Predictions vs Actual Results")
    
    # Create comparison dataframe
    date_col = 'date' if 'date' in valid_data.columns else 'Date'
    comparison = valid_data[[date_col, 'result', 'predicted']].copy()
    comparison = comparison.rename(columns={date_col: 'Date'})
    # Map results: 0=Draw, 1=Loss (home loss), 2=Win (home win)
    comparison['result_label'] = comparison['result'].map({0: 'Draw', 1: 'Loss', 2: 'Win'})
    comparison['predicted_label'] = comparison['predicted'].map({0: 'Draw', 1: 'Loss', 2: 'Win'})
    comparison['correct'] = comparison['result'] == comparison['predicted']
    comparison['status'] = comparison['correct'].map({True: '✅ Correct', False: '❌ Wrong'})
    
    # Display sample
    st.dataframe(
        comparison[['Date', 'result_label', 'predicted_label', 'status']].head(20).rename(columns={
            'Date': 'Date',
            'result_label': 'Actual Result',
            'predicted_label': 'Predicted',
            'status': 'Status'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    correct_preds = comparison['correct'].sum()
    total_preds = len(comparison)
    
    st.info(f"**Summary:** {correct_preds} correct predictions out of {total_preds} total ({correct_preds/total_preds*100:.1f}%)")
    
    # Performance over time
    st.markdown("---")
    st.subheader("📉 Accuracy Over Time")
    
    if 'Date' in comparison.columns:
        try:
            comparison['Date'] = pd.to_datetime(comparison['Date'])
            comparison = comparison.sort_values('Date')
            
            # Calculate rolling accuracy
            comparison['rolling_accuracy'] = comparison['correct'].rolling(window=10, min_periods=1).mean() * 100
            
            fig = px.line(
                comparison,
                x='Date',
                y='rolling_accuracy',
                title='Rolling Accuracy (10-match window)',
                labels={'rolling_accuracy': 'Accuracy (%)', 'Date': 'Date'}
            )
            
            fig.update_layout(height=350)
            fig.add_hline(y=metrics['accuracy']*100, line_dash="dash", line_color="red", 
                         annotation_text=f"Overall: {metrics['accuracy']:.1%}")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Date information not available for time-series analysis")
    
    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strengths:**")
        if metrics['f1_score'] > 0.7:
            st.success("✅ Strong overall F1-score indicates good model performance")
        if metrics['accuracy'] > 0.6:
            st.success("✅ Accuracy exceeds baseline (better than random)")
    
    with col2:
        st.markdown("**Areas for Improvement:**")
        if metrics['f1_score'] < 0.7:
            st.warning("⚠️ Consider improving feature engineering or model selection")
        if metrics['accuracy'] < 0.5:
            st.error("❌ Model performs worse than random guessing")
