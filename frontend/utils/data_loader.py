"""
Data loading utilities for the dashboard
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, Optional

class DataLoader:
    """Class to handle all data loading operations"""
    
    def __init__(self):
        # Get the data directory path
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        
    def load_matches_data(self) -> Optional[pd.DataFrame]:
        """Load the main matches dataset"""
        try:
            file_path = self.data_dir / "pl_matches_final_cleaned.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
            else:
                print(f"File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading matches data: {e}")
            return None
    
    def load_teams_data(self) -> Optional[pd.DataFrame]:
        """Load the teams dataset"""
        try:
            file_path = self.data_dir / "pl_teams.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
            else:
                print(f"File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading teams data: {e}")
            return None
    
    def load_current_matches(self) -> Optional[pd.DataFrame]:
        """Load current matches dataset"""
        try:
            file_path = self.data_dir / "currmatches.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
            else:
                print(f"File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading current matches: {e}")
            return None
    
    def load_premier_league_matches(self) -> Optional[pd.DataFrame]:
        """Load Premier League 2025-2026 matches"""
        try:
            file_path = self.data_dir / "premier_league_matches_2025_2026.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
            else:
                print(f"File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading premier league matches: {e}")
            return None
    
    def get_team_list(self) -> list:
        """
        Get list of teams that have data in currmatches.csv
        Only returns teams with home_team_ and away_team_ columns
        """
        # Load currmatches to see which teams have actual data
        curr_matches = self.load_current_matches()
        if curr_matches is None:
            return []
        
        # Extract team names from home_team_ columns
        home_cols = [col for col in curr_matches.columns if col.startswith('home_team_')]
        teams = [col.replace('home_team_', '') for col in home_cols]
        
        return sorted(teams)
    
    def get_match_results_summary(self) -> Dict:
        """Get summary statistics of match results"""
        df = self.load_matches_data()
        if df is None or 'home_result' not in df.columns:
            return {}
        
        result_counts = df['home_result'].value_counts()
        total_matches = len(df)
        
        summary = {
            'total_matches': total_matches,
            'home_wins': int(result_counts.get(2, 0)),
            'draws': int(result_counts.get(1, 0)),
            'away_wins': int(result_counts.get(0, 0)),
            'home_win_percentage': (result_counts.get(2, 0) / total_matches * 100) if total_matches > 0 else 0,
            'draw_percentage': (result_counts.get(1, 0) / total_matches * 100) if total_matches > 0 else 0,
            'away_win_percentage': (result_counts.get(0, 0) / total_matches * 100) if total_matches > 0 else 0,
        }
        
        return summary
    
    def get_team_stats(self, team_name: str) -> Dict:
        """Get statistics for a specific team"""
        df = self.load_matches_data()
        if df is None:
            return {}
        
        # Get all columns that contain team names
        home_team_cols = [col for col in df.columns if col.startswith('home_team_')]
        away_team_cols = [col for col in df.columns if col.startswith('away_team_')]
        
        # Find matches where this team played
        home_matches = df[df[f'home_team_{team_name}'] == True] if f'home_team_{team_name}' in df.columns else pd.DataFrame()
        away_matches = df[df[f'away_team_{team_name}'] == True] if f'away_team_{team_name}' in df.columns else pd.DataFrame()
        
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return {'error': f'No matches found for {team_name}'}
        
        # Calculate wins, draws, losses
        # Encoding: home_result: 0=home loss (away win), 1=draw, 2=home win
        home_wins = len(home_matches[home_matches['home_result'] == 2])
        home_draws = len(home_matches[home_matches['home_result'] == 1])
        home_losses = len(home_matches[home_matches['home_result'] == 0])  # home loss = away win
        
        # When team plays away, reverse the logic
        away_wins = len(away_matches[away_matches['home_result'] == 0])  # home_result=0 means home lost, away won
        away_draws = len(away_matches[away_matches['home_result'] == 1])
        away_losses = len(away_matches[away_matches['home_result'] == 2])  # home_result=2 means home won, away lost
        
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses
        
        stats = {
            'team_name': team_name,
            'total_matches': total_matches,
            'wins': total_wins,
            'draws': total_draws,
            'losses': total_losses,
            'win_rate': (total_wins / total_matches * 100) if total_matches > 0 else 0,
            'home_matches': len(home_matches),
            'away_matches': len(away_matches),
            'avg_home_elo': home_matches['home_elo'].mean() if len(home_matches) > 0 else 0,
            'avg_away_elo': away_matches['away_elo'].mean() if len(away_matches) > 0 else 0,
        }
        
        return stats
    
    def get_recent_matches(self, limit: int = 20) -> Optional[pd.DataFrame]:
        """Get most recent matches"""
        df = self.load_matches_data()
        if df is None or 'date' not in df.columns:
            return None
        
        # Sort by date and get the most recent matches
        df_sorted = df.sort_values('date', ascending=False)
        return df_sorted.head(limit)
    
    def get_team_recent_form(self, team_name: str, num_matches: int = 5) -> Dict:
        """
        Calculate team's recent form statistics based on last N matches
        Returns features needed for model prediction
        """
        df = self.load_current_matches()
        if df is None or 'Date' not in df.columns:
            return {}
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Get home team columns
        home_team_cols = [col for col in df.columns if col.startswith('home_team_')]
        away_team_cols = [col for col in df.columns if col.startswith('away_team_')]
        
        # Find matches where team played
        team_col = f'home_team_{team_name}'
        away_col = f'away_team_{team_name}'
        
        home_matches = df[df[team_col] == True].copy() if team_col in df.columns else pd.DataFrame()
        away_matches = df[df[away_col] == True].copy() if away_col in df.columns else pd.DataFrame()
        
        # Calculate stats from recent home matches
        home_stats = {}
        if len(home_matches) > 0:
            recent_home = home_matches.tail(num_matches)
            home_stats = {
                'avg_gf': recent_home['home_avg_gf_5'].mean() if 'home_avg_gf_5' in recent_home.columns else 1.5,
                'avg_ga': recent_home['home_avg_ga_5'].mean() if 'home_avg_ga_5' in recent_home.columns else 1.5,
                'avg_pts': recent_home['home_avg_pts_5'].mean() if 'home_avg_pts_5' in recent_home.columns else 1.5,
                'form_score': recent_home['home_form_score'].mean() if 'home_form_score' in recent_home.columns else 0,
                'elo': recent_home['home_elo'].iloc[-1] if 'home_elo' in recent_home.columns else 1500,
                'xg': recent_home['home_xg'].mean() if 'home_xg' in recent_home.columns else 1.5,
            }
        else:
            home_stats = {
                'avg_gf': 1.5, 'avg_ga': 1.5, 'avg_pts': 1.5,
                'form_score': 0, 'elo': 1500, 'xg': 1.5
            }
        
        # Calculate stats from recent away matches
        away_stats = {}
        if len(away_matches) > 0:
            recent_away = away_matches.tail(num_matches)
            away_stats = {
                'avg_gf': recent_away['away_avg_gf_5'].mean() if 'away_avg_gf_5' in recent_away.columns else 1.5,
                'avg_ga': recent_away['away_avg_ga_5'].mean() if 'away_avg_ga_5' in recent_away.columns else 1.5,
                'avg_pts': recent_away['away_avg_pts_5'].mean() if 'away_avg_pts_5' in recent_away.columns else 1.5,
                'form_score': recent_away['away_form_score'].mean() if 'away_form_score' in recent_away.columns else 0,
                'elo': recent_away['away_elo'].iloc[-1] if 'away_elo' in recent_away.columns else 1500,
                'xg': recent_away['away_xg'].mean() if 'away_xg' in recent_away.columns else 1.5,
            }
        else:
            away_stats = {
                'avg_gf': 1.5, 'avg_ga': 1.5, 'avg_pts': 1.5,
                'form_score': 0, 'elo': 1500, 'xg': 1.5
            }
        
        return {
            'home': home_stats,
            'away': away_stats
        }
    
    def prepare_prediction_features(self, home_team: str, away_team: str) -> Optional[pd.DataFrame]:
        """
        Prepare feature vector for model prediction
        Returns DataFrame with same features as training data
        """
        # Get recent form for both teams
        home_form = self.get_team_recent_form(home_team)
        away_form = self.get_team_recent_form(away_team)
        
        if not home_form or not away_form:
            return None
        
        # Get home team stats when playing at home
        home_stats = home_form.get('home', {})
        # Get away team stats when playing away
        away_stats = away_form.get('away', {})
        
        # Load current matches to get all feature columns
        df = self.load_current_matches()
        if df is None:
            return None
        
        # Get all team columns from the dataset
        home_team_cols = [col for col in df.columns if col.startswith('home_team_')]
        away_team_cols = [col for col in df.columns if col.startswith('away_team_')]
        
        # Create feature dictionary with numeric features
        features = {
            'home_xg': home_stats.get('xg', 1.5),
            'away_xg': away_stats.get('xg', 1.5),
            'home_avg_gf_5': home_stats.get('avg_gf', 1.5),
            'home_avg_ga_5': home_stats.get('avg_ga', 1.5),
            'home_avg_pts_5': home_stats.get('avg_pts', 1.5),
            'home_form_score': home_stats.get('form_score', 0),
            'away_avg_gf_5': away_stats.get('avg_gf', 1.5),
            'away_avg_ga_5': away_stats.get('avg_ga', 1.5),
            'away_avg_pts_5': away_stats.get('avg_pts', 1.5),
            'away_form_score': away_stats.get('form_score', 0),
            'home_elo': home_stats.get('elo', 1500),
            'away_elo': away_stats.get('elo', 1500),
        }
        
        # Add one-hot encoded team columns
        for col in home_team_cols:
            team_name = col.replace('home_team_', '')
            features[col] = (team_name == home_team)
        
        for col in away_team_cols:
            team_name = col.replace('away_team_', '')
            features[col] = (team_name == away_team)
        
        # Convert to DataFrame
        return pd.DataFrame([features])
