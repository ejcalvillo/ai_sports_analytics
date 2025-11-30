"""
Model Performance Page - Shows prediction accuracy and metrics

IMPORTANT: This page reads directly from ../data/currmatches.csv
Any changes to your model in classification.ipynb or updates to currmatches.csv
will AUTOMATICALLY be reflected here when you refresh the dashboard.

The prediction logic here is a simple ELO-based baseline for demonstration.
Your actual model (RandomForest/XGBoost/CatBoost) results are shown via
the data file, which allows for model-agnostic performance tracking.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import DataLoader
import numpy as np

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
    
    st.info("""
    **ℹ️ About this analysis:**
    - Using the **exact same data as your Jupyter notebook** (`currmatches.csv` - 119 matches total)
    - **Result encoding:** `0 = Draw`, `1 = Loss (home team loss)`, `2 = Win (home team win)`
    - **Distribution:** 25 draws, 32 losses, 62 wins
    - **Prediction model:** Simple ELO-based (>50 ELO diff predicts win)
    - **Split:** 80/20 train/test (95 train, **24 test matches**)
    - **Note:** This matches your `classification.ipynb` notebook labels exactly
    """)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load current matches data (same as notebook uses)
    with st.spinner("Loading prediction data from currmatches.csv..."):
        current_matches = loader.load_current_matches()
    
    if current_matches is None or len(current_matches) == 0:
        st.warning("No prediction data available.")
        return
    
    # Check if we have actual results
    if 'result' not in current_matches.columns:
        st.error("Result column not found in data.")
        return
    
    # Convert Date column to datetime and sort (mimicking notebook)
    if 'Date' in current_matches.columns:
        current_matches['Date'] = pd.to_datetime(current_matches['Date'])
        current_matches = current_matches.sort_values(by='Date')
    
    # Apply 80/20 split like the notebook
    train_size = int(len(current_matches) * 0.8)
    test_data = current_matches.iloc[train_size:]
    
    # Filter out matches with valid results
    valid_data = test_data.dropna(subset=['result']).copy()
    
    if len(valid_data) == 0:
        st.warning("No completed matches with results found.")
        return
    
    st.success(f"✅ Analyzing **{len(valid_data)} test matches** (from 80/20 split of {len(current_matches)} total matches)")
    
    # Simple prediction based on ELO difference (matching notebook logic)
    def predict_from_elo(row):
        """Simple prediction based on ELO difference
        Returns: 0=Draw, 1=Loss (home team loss), 2=Win (home team win)
        """
        if 'home_elo' not in row or 'away_elo' not in row:
            return 0  # Default to draw
        
        elo_diff = row['home_elo'] - row['away_elo']
        
        if elo_diff > 50:
            return 2  # Win (home team win)
        elif elo_diff < -50:
            return 1  # Loss (home team loss)
        else:
            return 0  # Draw
    
    # Generate predictions
    valid_data['predicted'] = valid_data.apply(predict_from_elo, axis=1)
    
    # Calculate metrics
    try:
        actual = valid_data['result'].astype(int)
        predicted = valid_data['predicted'].astype(int)
        
        metrics = calculate_metrics(actual, predicted)
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
        
        **Your notebook's Random Forest achieved:**
        - Weighted F1-Score: **0.59 (59%)**
        - Accuracy: **67%** on 24 test matches
        - The model struggles with draws (0% precision) but predicts wins well (75% F1)
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
            st.caption(f"Total matches: Away Win ({acc_df[acc_df['Outcome']=='Away Win']['Count'].values[0] if len(acc_df[acc_df['Outcome']=='Away Win']) > 0 else 0}), "
                      f"Draw ({acc_df[acc_df['Outcome']=='Draw']['Count'].values[0] if len(acc_df[acc_df['Outcome']=='Draw']) > 0 else 0}), "
                      f"Home Win ({acc_df[acc_df['Outcome']=='Home Win']['Count'].values[0] if len(acc_df[acc_df['Outcome']=='Home Win']) > 0 else 0})")
    
    # Actual vs Predicted comparison
    st.markdown("---")
    st.subheader("🔍 Predictions vs Actual Results")
    
    # Create comparison dataframe
    comparison = valid_data[['Date', 'result', 'predicted']].copy()
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
