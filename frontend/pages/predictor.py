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
def load_model():
    """Load the trained CatBoost model and feature names"""
    try:
        base_dir = Path(__file__).parent.parent.parent
        models_dir = base_dir / "models"
        
        model_path = models_dir / "catboost_model.pkl"
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
                         loader: DataLoader) -> dict:
    """
    Predict match outcome using the trained CatBoost model
    
    Args:
        home_team: Name of home team
        away_team: Name of away team  
        home_stats: Dictionary of home team statistics
        away_stats: Dictionary of away team statistics
        loader: DataLoader instance for feature preparation
        
    Returns:
        Dictionary with prediction results
    """
    # Load the model
    model, feature_names = load_model()
    
    if model is None:
        # Fallback to simple ELO-based prediction
        home_strength = home_stats.get('avg_home_elo', 1500)
        away_strength = away_stats.get('avg_away_elo', 1500)
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
            'model_used': 'ELO Fallback'
        }
    
    # Prepare features for prediction
    features_df = loader.prepare_prediction_features(home_team, away_team)
    
    if features_df is None:
        st.error("Unable to prepare prediction features")
        return None
    
    # Ensure features are in the same order as training
    if feature_names:
        # Reorder columns to match training data
        missing_cols = [col for col in feature_names if col not in features_df.columns]
        if missing_cols:
            # Add missing columns with default values
            for col in missing_cols:
                features_df[col] = False
        
        features_df = features_df[feature_names]
    
    # Make prediction
    pred = model.predict(features_df)
    # CatBoost may return 2D array, flatten if needed
    prediction_class = pred.flatten()[0] if len(pred.shape) > 1 else pred[0]
    
    proba = model.predict_proba(features_df)
    # CatBoost returns probabilities in correct format
    prediction_proba = proba[0] if len(proba.shape) > 1 else proba
    
    # Class encoding: 0=Draw, 1=Loss (home loss), 2=Win (home win)
    draw_prob = prediction_proba[0]
    loss_prob = prediction_proba[1]  # home loss = away win
    win_prob = prediction_proba[2]   # home win
    
    # Determine prediction text
    if prediction_class == 2:
        prediction = "Home Win"
    elif prediction_class == 1:
        prediction = "Away Win"
    else:
        prediction = "Draw"
    
    # Get ELO for display
    home_form = loader.get_team_recent_form(home_team)
    away_form = loader.get_team_recent_form(away_team)
    home_strength = home_form.get('home', {}).get('elo', 1500)
    away_strength = away_form.get('away', {}).get('elo', 1500)
    elo_diff = home_strength - away_strength
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'prediction': prediction,
        'home_win_probability': round(win_prob * 100, 1),
        'draw_probability': round(draw_prob * 100, 1),
        'away_win_probability': round(loss_prob * 100, 1),
        'confidence': round(max(win_prob, draw_prob, loss_prob) * 100, 1),
        'home_strength': home_strength,
        'away_strength': away_strength,
        'elo_difference': round(elo_diff, 1),
        'model_used': 'CatBoost (1500 iterations, 71% accuracy)'
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
    st.markdown('<div class="sub-header">🔮 Match Predictor</div>', unsafe_allow_html=True)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Get team list
    teams = loader.get_team_list()
    
    if not teams:
        st.error("Unable to load team data.")
        return
    
    # Main prediction interface
    st.markdown("### ⚽ Select Teams for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "🏠 Home Team:",
            options=teams,
            index=0,
            key='home_team'
        )
    
    with col2:
        away_team = st.selectbox(
            "✈️ Away Team:",
            options=teams,
            index=min(1, len(teams)-1),
            key='away_team'
        )
    
    # Predict button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("⚠️ Please select two different teams.")
        else:
            with st.spinner("Analyzing teams and generating prediction..."):
                # Get team statistics
                home_stats = loader.get_team_stats(home_team)
                away_stats = loader.get_team_stats(away_team)
                
                if 'error' in home_stats or 'error' in away_stats:
                    st.error("Unable to retrieve team statistics.")
                    return
                
                # Make prediction using trained CatBoost model
                prediction = predict_match_outcome(
                    home_team, away_team,
                    home_stats, away_stats,
                    loader
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
            
            # Key factors
            st.markdown("---")
            st.subheader("📈 Key Factors")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{home_team}**")
                st.write(f"- ELO Rating: **{prediction['home_strength']:.0f}**")
                st.write(f"- Win Rate: **{home_stats.get('win_rate', 0):.1f}%**")
                st.write(f"- Total Wins: **{home_stats.get('wins', 0)}**")
            
            with col2:
                st.markdown(f"**{away_team}**")
                st.write(f"- ELO Rating: **{prediction['away_strength']:.0f}**")
                st.write(f"- Win Rate: **{away_stats.get('win_rate', 0):.1f}%**")
                st.write(f"- Total Wins: **{away_stats.get('wins', 0)}**")
            
            # Radar Chart Comparison and Recent Form
            st.markdown("### 🎯 Team Stats Comparison")
            
            # Create two columns for side-by-side charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### 📊 Stats Pentagon")
                
                # Prepare data for radar chart
                categories = ['ELO Rating', 'Goals For', 'Goals Against', 'Form Score', 'Avg Points']
                
                # Normalize values to 0-100 scale for better visualization
                def normalize(value, min_val, max_val):
                    if max_val == min_val:
                        return 50
                    return ((value - min_val) / (max_val - min_val)) * 100
                
                # Get raw values for both teams
                home_elo = prediction['home_strength']
                away_elo = prediction['away_strength']
                home_gf = home_form.get('home', {}).get('avg_gf', 1.5)
                away_gf = away_form.get('away', {}).get('avg_gf', 1.5)
                home_ga = home_form.get('home', {}).get('avg_ga', 1.5)
                away_ga = away_form.get('away', {}).get('avg_ga', 1.5)
                home_form_score = home_form.get('home', {}).get('form_score', 0)
                away_form_score = away_form.get('away', {}).get('form_score', 0)
                home_pts = home_form.get('home', {}).get('avg_pts', 1.5)
                away_pts = away_form.get('away', {}).get('avg_pts', 1.5)
                
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
                
                # Normalize Form Score (typical range: -5 to 15)
                form_min, form_max = -5, 15
                home_form_norm = normalize(home_form_score, form_min, form_max)
                away_form_norm = normalize(away_form_score, form_min, form_max)
                
                # Normalize Points (typical range: 0-3)
                pts_min, pts_max = 0, 3
                home_pts_norm = normalize(home_pts, pts_min, pts_max)
                away_pts_norm = normalize(away_pts, pts_min, pts_max)
                
                # Create radar chart
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[home_elo_norm, home_gf_norm, home_ga_norm, home_form_norm, home_pts_norm],
                    theta=categories,
                    fill='toself',
                    name=home_team,
                    line_color='#2ca02c',
                    fillcolor='rgba(44, 160, 44, 0.3)'
                ))
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[away_elo_norm, away_gf_norm, away_ga_norm, away_form_norm, away_pts_norm],
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
                - **ELO**: Team strength rating
                - **Goals For**: Avg goals scored
                - **Goals Against**: Defense (inverted)
                - **Form Score**: Recent trend
                - **Avg Points**: Points per match
                """)
            
            with chart_col2:
                st.markdown("#### 📊 Last 5 Matches Form")
                
                # Get last 5 results for both teams
                home_results = loader.get_team_last_n_results(home_team, 5)
                away_results = loader.get_team_last_n_results(away_team, 5)
                
                # Count wins, draws, losses for both teams
                home_wins = home_results.count('W')
                home_draws = home_results.count('D')
                home_losses = home_results.count('L')
                away_wins = away_results.count('W')
                away_draws = away_results.count('D')
                away_losses = away_results.count('L')
                
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
                
                # Detailed match-by-match breakdown
                st.markdown("**Match-by-Match Results:**")
                
                # Create visual indicators for each team
                def result_emoji(result):
                    if result == 'W':
                        return '✅'
                    elif result == 'D':
                        return '🟨'
                    elif result == 'L':
                        return '❌'
                    else:
                        return '⚪'
                
                # Pad results if needed
                while len(home_results) < 5:
                    home_results.insert(0, '-')
                while len(away_results) < 5:
                    away_results.insert(0, '-')
                
                # Display in columns
                match_col1, match_col2 = st.columns(2)
                
                with match_col1:
                    st.markdown(f"**{home_team}**")
                    home_display = ' '.join([result_emoji(r) for r in home_results])
                    st.markdown(f"{home_display}")
                    st.caption(f"{home_wins}W-{home_draws}D-{home_losses}L")
                
                with match_col2:
                    st.markdown(f"**{away_team}**")
                    away_display = ' '.join([result_emoji(r) for r in away_results])
                    st.markdown(f"{away_display}")
                    st.caption(f"{away_wins}W-{away_draws}D-{away_losses}L")
                
                st.caption("*From last 5 matches (oldest → newest)*")
            
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
            🤖 **AI Model:** {prediction.get('model_used', 'CatBoost')}
            
            This prediction is generated by the trained machine learning model from `classification.ipynb`.
            The model was trained on {119} Premier League matches using features like:
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
