"""
Match Predictor Page - Main prediction interface

Uses the trained CatBoost model from classification.ipynb (71% accuracy)
Predictions are made using the same features as the training data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path
from utils.data_loader import DataLoader

# Load the trained model (cache it so we don't reload on every prediction)
@st.cache_resource
def load_model(model_name: str = "catboost"):
    """Load a trained model and feature names"""
    try:
        # From frontend/pages/predictor.py, parents[2] = ai_sports_analytics/
        base_dir = Path(__file__).resolve().parents[2]
        models_dir = base_dir / "models"
        
        # Map model names to filenames
        model_files = {
            "catboost": "catboost_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "xgboost": "xgboost_model.pkl"
        }
        
        if model_name not in model_files:
            st.error(f"❌ Unknown model: {model_name}. Available models: {', '.join(model_files.keys())}")
            return None, None
        
        model_path = models_dir / model_files[model_name]
        features_path = models_dir / "feature_names.pkl"
        
        if not model_path.exists():
            st.error(f"❌ Model not found at {model_path}. Please run the classification notebook first.")
            return None, None
        
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path) if features_path.exists() else None
        
        return model, feature_names
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def predict_match_outcome(home_team: str, away_team: str, 
                         home_stats: dict, away_stats: dict,
                         loader: DataLoader,
                         use_unseen: bool = False) -> dict:
    """
    Predict match outcome using the trained Random Forest model (matches predictmatch.ipynb logic)
    
    Args:
        home_team: Name of home team
        away_team: Name of away team  
        home_stats: Dictionary of home team statistics (used if use_unseen=False)
        away_stats: Dictionary of away team statistics (used if use_unseen=False)
        loader: DataLoader instance for feature preparation
        use_unseen: If True, use features for unseen matches from pl25-26.csv
        
    Returns:
        Dictionary with prediction results
    """
    # Load the Random Forest model
    model, feature_names = load_model("random_forest")
    
    if model is None:
        # Fallback to simple ELO-based prediction
        home_strength = home_stats.get('home_elo_before', 1500)
        away_strength = away_stats.get('away_elo_before', 1500)
        elo_diff = home_strength - away_strength
        
        if elo_diff > 50:
            home_win_prob = 0.60 + (min(elo_diff, 200) / 400) * 0.20
        elif elo_diff < -50:
            home_win_prob = 0.40 - (min(abs(elo_diff), 200) / 400) * 0.20
        else:
            home_win_prob = 0.50
        
        draw_prob = 0.25
        away_win_prob = 1.0 - home_win_prob - draw_prob
        
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        if home_win_prob > draw_prob and home_win_prob > away_win_prob:
            prediction = "Home Win"
        elif away_win_prob > draw_prob and away_win_prob > home_win_prob:
            prediction = "Away Win"
        else:
            prediction = "Draw"
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction,
            'home_win_probability': round(home_win_prob * 100, 1),
            'draw_probability': round(draw_prob * 100, 1),
            'away_win_probability': round(away_win_prob * 100, 1),
            'confidence': round(max(home_win_prob, draw_prob, away_win_prob) * 100, 1),
            'home_strength': home_strength,
            'away_strength': away_strength,
            'elo_difference': round(elo_diff, 1),
            'model_used': 'Random Forest Fallback'
        }
    
    # Prepare features for prediction
    if use_unseen:
        features_df = loader.prepare_unseen_match_features(home_team, away_team)
    else:
        features_df = loader.prepare_prediction_features(home_team, away_team)
    
    if features_df is None:
        st.error("Unable to prepare prediction features")
        return None
    
    # Drop team names if they exist (model doesn't use them)
    X = features_df.drop(columns=['home_team', 'away_team'], errors='ignore')
    
    # Make prediction
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    
    # Class encoding from notebook: 0=Away win, 1=Draw, 2=Home win
    away_win_prob = proba[0]
    draw_prob = proba[1]
    home_win_prob = proba[2]
    
    # Determine prediction text
    if pred == 2:
        prediction = "Home Win"
    elif pred == 1:
        prediction = "Draw"
    else:
        prediction = "Away Win"
    
    # Get ELO for display (extract from DataFrame)
    try:
        if 'home_elo_before' in features_df.columns:
            home_strength = float(features_df.iloc[0]['home_elo_before'])
        else:
            home_strength = 1500
        
        if 'away_elo_before' in features_df.columns:
            away_strength = float(features_df.iloc[0]['away_elo_before'])
        else:
            away_strength = 1500
        
        elo_diff = home_strength - away_strength
    except Exception as e:
        home_strength = 1500
        away_strength = 1500
        elo_diff = 0
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'prediction': prediction,
        'home_win_probability': round(home_win_prob * 100, 1),
        'draw_probability': round(draw_prob * 100, 1),
        'away_win_probability': round(away_win_prob * 100, 1),
        'confidence': round(max(home_win_prob, draw_prob, away_win_prob) * 100, 1),
        'home_strength': home_strength if isinstance(home_strength, (int, float)) else 1500,
        'away_strength': away_strength if isinstance(away_strength, (int, float)) else 1500,
        'elo_difference': round(elo_diff if isinstance(elo_diff, (int, float)) else 0, 1),
        'model_used': 'Random Forest (Trained on 119 Premier League matches)',
        'is_unseen': use_unseen
    }

def calculate_predicted_score(prediction: dict, home_form: dict, away_form: dict) -> tuple:
    """
    Calculate predicted integer score based on team form and match outcome
    Uses average goals scored/conceded from recent matches
    Returns whole number scores only - no decimals allowed in soccer!
    """
    outcome = prediction['prediction']
    
    # Get average goals from recent form (home stats when playing home, away stats when playing away)
    home_avg_goals = home_form.get('home', {}).get('xg', 1.5)  # Expected goals
    away_avg_goals = away_form.get('away', {}).get('xg', 1.2)  # Expected goals
    
    # Adjust scores based on predicted outcome
    if outcome == "Home Win":
        # Home team expected to score more
        home_score = round(max(home_avg_goals * 1.2, 2))  # Boost by 20% for win
        away_score = round(max(away_avg_goals * 0.8, 0))  # Reduce by 20%
    elif outcome == "Away Win":
        # Away team expected to score more
        home_score = round(max(home_avg_goals * 0.8, 0))  # Reduce by 20%
        away_score = round(max(away_avg_goals * 1.2, 2))  # Boost by 20% for win
    else:  # Draw
        # Both teams score similar amounts
        home_score = round(home_avg_goals)
        away_score = round(away_avg_goals)
        # Ensure it's actually a draw
        if home_score != away_score:
            avg = (home_score + away_score) // 2
            home_score = away_score = max(avg, 1)
    
    # Ensure realistic score range (0-5 goals typical)
    home_score = max(0, min(int(home_score), 5))
    away_score = max(0, min(int(away_score), 5))
    
    return (home_score, away_score)

def show():
    """Display the match predictor page"""
    
    # Initialize data loader
    loader = DataLoader()
    
    # Get team list
    teams = loader.load_teams_data()
    
    if not teams:
        st.error("Unable to load team data.")
        return
    
    # Only allow prediction on unseen matches
    st.markdown("### 📅 Predict Upcoming Matches (2025-26 Season)")
    
    unseen_matches = loader.get_unseen_matches(limit=10)
    if unseen_matches:
        # Initialize session state for selected match
        if 'selected_match_idx' not in st.session_state:
            st.session_state.selected_match_idx = None
        
        # Display matches as grid buttons
        st.markdown("**Select a match to predict:**")
        
        # Add custom CSS for active button styling
        st.markdown("""
        <style>
        .match-button-active {
            border: 3px solid #1f77b4 !important;
            background-color: #e8f0f7 !important;
            box-shadow: 0 0 10px rgba(31, 119, 180, 0.5);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a grid of match buttons (2 columns)
        cols = st.columns(2)
        for idx, (home_team_opt, away_team_opt) in enumerate(unseen_matches):
            col = cols[idx % 2]
            
            with col:
                # Check if this button is selected
                is_selected = st.session_state.selected_match_idx == idx
                
                # Create button with selected state indicator
                button_label = f"🏠 {home_team_opt}\nvs\n✈️ {away_team_opt}"
                if is_selected:
                    button_label = f"✓ 🏠 {home_team_opt}\nvs\n✈️ {away_team_opt}"
                
                button_clicked = st.button(
                    button_label,
                    key=f"match_btn_{idx}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                )
                
                if button_clicked:
                    st.session_state.selected_match_idx = idx
                    st.rerun()
        
        # Show selected match and prediction button
        if st.session_state.selected_match_idx is not None:
            home_team, away_team = unseen_matches[st.session_state.selected_match_idx]
            st.info(f"🔮 Selected: **{home_team}** vs **{away_team}**")
            
            # Predict button
            if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
                if home_team == away_team:
                    st.error("⚠️ Please select two different teams.")
                else:
                    with st.spinner("Analyzing teams and generating prediction..."):
                        # For unseen matches, we don't need historical stats
                        home_stats = {}
                        away_stats = {}
                        
                        # Make prediction using Random Forest model
                        prediction = predict_match_outcome(
                            home_team, away_team,
                            home_stats, away_stats,
                            loader,
                            use_unseen=True
                        )
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    st.markdown("---")
                    
                    # Main prediction result
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.markdown(f"<h3 style='text-align: center;'>{home_team}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; color: #888;'>ELO: {prediction['home_strength']:.0f}</p>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>VS</h1>", unsafe_allow_html=True)
                        
                        # Prediction result
                        outcome_text = prediction['prediction']
                        if outcome_text == "Home Win":
                            outcome_color = "#2ca02c"
                            outcome_icon = "🏆"
                        elif outcome_text == "Away Win":
                            outcome_color = "#d62728"
                            outcome_icon = "🏆"
                        else:
                            outcome_color = "#ff7f0e"
                            outcome_icon = "🤝"
                        
                        st.markdown(f"<h2 style='text-align: center; color: {outcome_color};'>{outcome_icon} {outcome_text}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; font-size: 1.2em;'>Confidence: <strong>{prediction['confidence']}%</strong></p>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"<h3 style='text-align: center;'>{away_team}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; color: #888;'>ELO: {prediction['away_strength']:.0f}</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Probability breakdown
                    st.subheader("📊 Win Probabilities")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label=f"🏠 {home_team} Win",
                            value=f"{prediction['home_win_probability']}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="🤝 Draw",
                            value=f"{prediction['draw_probability']}%"
                        )
                    
                    with col3:
                        st.metric(
                            label=f"✈️ {away_team} Win",
                            value=f"{prediction['away_win_probability']}%"
                        )
                    
                    # Visualization
                    fig = go.Figure()
                    
                    outcomes = ['Home Win', 'Draw', 'Away Win']
                    probabilities = [
                        prediction['home_win_probability'],
                        prediction['draw_probability'],
                        prediction['away_win_probability']
                    ]
                    colors = ['#2ca02c', '#ff7f0e', '#d62728']
                    
                    fig.add_trace(go.Bar(
                        x=outcomes,
                        y=probabilities,
                        text=[f"{p}%" for p in probabilities],
                        textposition='auto',
                        marker_color=colors,
                        hovertemplate='%{x}: %{y}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Probability Distribution',
                        xaxis_title='Outcome',
                        yaxis_title='Probability (%)',
                        yaxis_range=[0, 100],
                        showlegend=False,
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Predicted score (integers only - no decimals!)
                    # Get team form for score calculation
                    home_form = loader.get_team_recent_form(home_team)
                    away_form = loader.get_team_recent_form(away_team)
                    predicted_home, predicted_away = calculate_predicted_score(prediction, home_form, away_form)
                    
                    st.markdown("---")
                    st.subheader("⚽ Predicted Score")
                    
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"<h3 style='text-align: center;'>{home_team}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        # Display integer scores only (format as :d to ensure no decimals)
                        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{int(predicted_home)} - {int(predicted_away)}</h1>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"<h3 style='text-align: center;'>{away_team}</h3>", unsafe_allow_html=True)
                    
                    st.caption("🔢 Predicted scoreline based on match outcome prediction. Actual goals may vary.")
                    
                    # Retrieve form data upfront for all sections
                    home_form_data = loader.get_team_recent_form(home_team, 5)
                    away_form_data = loader.get_team_recent_form(away_team, 5)
                    
                    # Key factors
                    st.markdown("---")
                    st.subheader("📈 Key Factors (Last 5 Matches)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{home_team}**")
                        st.write(f"- ELO Rating: **{prediction['home_strength']:.0f}**")
                        home_form_wins = int(home_form_data.get('home', {}).get('wins', 0))
                        home_form_score = home_form_data.get('home', {}).get('form_score', 0)
                        st.write(f"- Form: **{home_form_wins}W-{int(home_form_data.get('home', {}).get('draws', 0))}D-{int(home_form_data.get('home', {}).get('losses', 0))}L** ({home_form_score:.0f} pts)")
                        st.write(f"- Avg Goals For: **{home_form_data.get('home', {}).get('avg_gf', 0):.2f}**")
                        st.write(f"- Avg Goals Against: **{home_form_data.get('home', {}).get('avg_ga', 0):.2f}**")
                        st.write(f"- Expected Goals (xG): **{home_form_data.get('home', {}).get('xg', 0):.2f}**")
                    
                    with col2:
                        st.markdown(f"**{away_team}**")
                        st.write(f"- ELO Rating: **{prediction['away_strength']:.0f}**")
                        away_form_wins = int(away_form_data.get('away', {}).get('wins', 0))
                        away_form_score = away_form_data.get('away', {}).get('form_score', 0)
                        st.write(f"- Form: **{away_form_wins}W-{int(away_form_data.get('away', {}).get('draws', 0))}D-{int(away_form_data.get('away', {}).get('losses', 0))}L** ({away_form_score:.0f} pts)")
                        st.write(f"- Avg Goals For: **{away_form_data.get('away', {}).get('avg_gf', 0):.2f}**")
                        st.write(f"- Avg Goals Against: **{away_form_data.get('away', {}).get('avg_ga', 0):.2f}**")
                        st.write(f"- Expected Goals (xG): **{away_form_data.get('away', {}).get('xg', 0):.2f}**")
                    
                    # Radar Chart Comparison and Recent Form
                    st.markdown("### 🎯 Team Stats Comparison")
                    
                    # Create two columns for side-by-side charts
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.markdown("#### 📊 Stats Pentagon")
                        
                        # Form data already retrieved above (ensures data is always available)
                        
                        # Prepare data for radar chart
                        categories = ['ELO Rating', 'Goals For', 'Goals Against', 'Form Score', 'Win Rate']
                        
                        # Normalize values to 0-100 scale for better visualization
                        def normalize(value, min_val, max_val):
                            if max_val == min_val:
                                return 50
                            return ((value - min_val) / (max_val - min_val)) * 100
                        
                        # Get raw values for both teams
                        home_elo = prediction['home_strength']
                        away_elo = prediction['away_strength']
                        home_gf = home_form_data.get('home', {}).get('avg_gf', 1.5)
                        away_gf = away_form_data.get('away', {}).get('avg_gf', 1.5)
                        home_ga = home_form_data.get('home', {}).get('avg_ga', 1.5)
                        away_ga = away_form_data.get('away', {}).get('avg_ga', 1.5)
                        home_form_score_chart = home_form_data.get('home', {}).get('form_score', 0)
                        away_form_score_chart = away_form_data.get('away', {}).get('form_score', 0)
                        home_win_rate = 0  # Not available for unseen matches, use 0
                        away_win_rate = 0  # Not available for unseen matches, use 0
                        
                        # Normalize ELO (typical range: 1200-2000)
                        elo_min, elo_max = 1200, 2000
                        home_elo_norm = normalize(home_elo, elo_min, elo_max)
                        away_elo_norm = normalize(away_elo, elo_min, elo_max)
                        
                        # Normalize Goals For (typical range: 0-3)
                        gf_min, gf_max = 0, 3
                        home_gf_norm = normalize(home_gf, gf_min, gf_max)
                        away_gf_norm = normalize(away_gf, gf_min, gf_max)
                        
                        # Normalize Goals Against (lower is better, so invert)
                        ga_min, ga_max = 0, 3
                        home_ga_norm = 100 - normalize(home_ga, ga_min, ga_max)
                        away_ga_norm = 100 - normalize(away_ga, ga_min, ga_max)
                        
                        # Normalize Form Score (typical range: 0-15 for 5 games)
                        form_min, form_max = 0, 15
                        home_form_norm = normalize(home_form_score_chart, form_min, form_max)
                        away_form_norm = normalize(away_form_score_chart, form_min, form_max)
                        
                        # Normalize Win Rate (typical range: 0-100)
                        wr_min, wr_max = 0, 100
                        home_wr_norm = normalize(home_win_rate, wr_min, wr_max)
                        away_wr_norm = normalize(away_win_rate, wr_min, wr_max)
                        
                        # Create radar chart
                        fig_radar = go.Figure()
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[home_elo_norm, home_gf_norm, home_ga_norm, home_form_norm, home_wr_norm],
                            theta=categories,
                            fill='toself',
                            name=home_team,
                            line_color='#2ca02c',
                            fillcolor='rgba(44, 160, 44, 0.3)'
                        ))
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[away_elo_norm, away_gf_norm, away_ga_norm, away_form_norm, away_wr_norm],
                            theta=categories,
                            fill='toself',
                            name=away_team,
                            line_color='#d62728',
                            fillcolor='rgba(214, 39, 40, 0.3)'
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100],
                                    showticklabels=True,
                                    ticks='',
                                    tickfont=dict(size=10)
                                )
                            ),
                            showlegend=True,
                            height=450,
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Add explanatory text
                        st.caption("""
                        **Pentagon Breakdown:**
                        - **ELO**: Team strength rating (1200-2000)
                        - **Goals For**: Avg goals scored (0-3)
                        - **Goals Against**: Defense quality inverted (0-3)
                        - **Form Score**: Last 5 matches (0-15 pts)
                        - **Win Rate**: Career win percentage (0-100%)
                        """)
                    
                    with chart_col2:
                        st.markdown("#### 📊 Team Form Summary (Last 5 Matches)")
                        
                        # Use form data already retrieved (avoid redundant retrieval)
                        home_wins = int(home_form_data.get('home', {}).get('wins', 0))
                        home_draws = int(home_form_data.get('home', {}).get('draws', 0))
                        home_losses = int(home_form_data.get('home', {}).get('losses', 0))
                        away_wins = int(away_form_data.get('away', {}).get('wins', 0))
                        away_draws = int(away_form_data.get('away', {}).get('draws', 0))
                        away_losses = int(away_form_data.get('away', {}).get('losses', 0))
                        
                        # Create grouped bar chart
                        fig_form = go.Figure()
                        
                        categories = ['Wins', 'Draws', 'Losses']
                        
                        fig_form.add_trace(go.Bar(
                            name=home_team,
                            x=categories,
                            y=[home_wins, home_draws, home_losses],
                            marker_color='#2ca02c',
                            text=[home_wins, home_draws, home_losses],
                            textposition='auto',
                            textfont=dict(size=14, color='white', family='Arial Black'),
                            hovertemplate='%{y} %{x}<extra></extra>'
                        ))
                        
                        fig_form.add_trace(go.Bar(
                            name=away_team,
                            x=categories,
                            y=[away_wins, away_draws, away_losses],
                            marker_color='#d62728',
                            text=[away_wins, away_draws, away_losses],
                            textposition='auto',
                            textfont=dict(size=14, color='white', family='Arial Black'),
                            hovertemplate='%{y} %{x}<extra></extra>'
                        ))
                        
                        fig_form.update_layout(
                            barmode='group',
                            xaxis=dict(
                                title='Result Type'
                            ),
                            yaxis=dict(
                                title='Number of Matches',
                                range=[0, 5.5],
                                dtick=1
                            ),
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.02,
                                xanchor='center',
                                x=0.5
                            ),
                            height=450,
                            margin=dict(l=60, r=40, t=60, b=60),
                            plot_bgcolor='rgba(240,240,240,0.3)'
                        )
                        
                        st.plotly_chart(fig_form, use_container_width=True)
                        
                        # Form summary
                        st.markdown("**Recent Form Summary (Last 5):**")
                        
                        # Display in columns
                        match_col1, match_col2 = st.columns(2)
                        
                        with match_col1:
                            st.markdown(f"**{home_team}**")
                            st.markdown(f"📊 {home_wins}W-{home_draws}D-{home_losses}L")
                            home_form_score = home_form_data.get('home', {}).get('form_score', 0)
                            st.markdown(f"⭐ Form: {home_form_score:.0f} pts")
                        
                        with match_col2:
                            st.markdown(f"**{away_team}**")
                            st.markdown(f"📊 {away_wins}W-{away_draws}D-{away_losses}L")
                            away_form_score = away_form_data.get('away', {}).get('form_score', 0)
                            st.markdown(f"⭐ Form: {away_form_score:.0f} pts")
                    
                    # ELO difference insight
                    elo_diff = abs(prediction['elo_difference'])
                    if elo_diff > 100:
                        strength_text = "**Strong advantage** for the higher-rated team"
                    elif elo_diff > 50:
                        strength_text = "**Moderate advantage** for the higher-rated team"
                    else:
                        strength_text = "**Very close match** - teams are evenly matched"
                    
                    st.info(f"**ELO Difference:** {elo_diff:.0f} points - {strength_text}")
                    
                    # Model info
                    st.markdown("---")
                    st.info(f"""
                    🤖 **AI Model:** {"Random Forest (Trained on 2021-2025 Premiere League Data.)"}
                    
                    This prediction is generated by the trained machine learning model from `classification.ipynb`.
                    The model was trained on {1640} Premier League matches using features like:
                    - Recent form (last 5 matches)
                    - Goals scored/conceded averages
                    - ELO ratings
                    - Expected goals (xG)
                    """)
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Note:** Predictions are based on historical data and statistical patterns. 
                    Actual results may vary due to injuries, form, weather, and other factors.
                    """)
