"""
Data loading utilities for the dashboard
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class DataLoader:
    """Class to handle all data loading operations"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.base_dir / "data"
        
    def load_matches_data(self) -> Optional[pd.DataFrame]:
        """Load the main matches dataset"""
        try:
            file_path = self.data_dir / "pl_final.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
            else:
                print(f"File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading matches data: {e}")
            return None
    
    def load_teams_data(self) -> list:
        """
        Returns list of unique teams based on home_team/away_team columns.
        """
        df = self.load_matches_data()
        if df is None:
            return []
        
        teams = pd.concat([df["home_team"], df["away_team"]]).unique().tolist()
        teams = [t for t in teams if isinstance(t, str)]
        teams.sort()
        return teams
    
    def get_match_results_summary(self) -> Dict:
        """Get summary statistics of match results"""
        df = self.load_matches_data()
        if df is None or "result" not in df.columns:
            return {}
        
        result_counts = df["result"].value_counts()
        total_matches = len(df)
        
        summary = {
            "total_matches": total_matches,
            "home_wins": int(result_counts.get(2, 0)),
            "draws": int(result_counts.get(1, 0)),
            "away_wins": int(result_counts.get(0, 0)),
            "home_win_percentage": (result_counts.get(2, 0) / total_matches * 100),
            "draw_percentage": (result_counts.get(1, 0) / total_matches * 100),
            "away_win_percentage": (result_counts.get(0, 0) / total_matches * 100),
        }
        
        return summary
    
    def get_team_stats(self, team_name: str) -> Dict:
        """Get statistics for a specific team"""
        df = self.load_matches_data()
        if df is None:
            return {}

        home_matches = df[df["home_team"] == team_name]
        away_matches = df[df["away_team"] == team_name]

        total_matches = len(home_matches) + len(away_matches)
        if total_matches == 0:
            return {"error": f"No matches found for {team_name}"}

        # result: 2 = home win, 1 = draw, 0 = away win
        home_wins = len(home_matches[home_matches["result"] == 2])
        home_draws = len(home_matches[home_matches["result"] == 1])
        home_losses = len(home_matches[home_matches["result"] == 0])

        away_wins = len(away_matches[away_matches["result"] == 0])
        away_draws = len(away_matches[away_matches["result"] == 1])
        away_losses = len(away_matches[away_matches["result"] == 2])

        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_losses = home_losses + away_losses

        stats = {
            "team_name": team_name,
            "total_matches": total_matches,
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_losses,
            "win_rate": total_wins / total_matches * 100,
        }
        
        return stats
    
    def get_recent_matches(self, limit: int = 20) -> Optional[pd.DataFrame]:
        df = self.load_matches_data()
        if df is None or "date" not in df.columns:
            return None
        
        df["date"] = pd.to_datetime(df["date"])
        df_sorted = df.sort_values("date", ascending=False)
        return df_sorted.head(limit)
    
    def get_team_recent_form(self, team_name: str, num_matches: int = 5) -> Dict:
        """
        Returns rolling stats for recent matches including wins/draws/losses.
        Works with the pl_final.csv rolling feature names.
        Result encoding: H=2 (home win), D=0 (draw), A=1 (away win)
        """
        df = self.load_matches_data()
        if df is None:
            return {}

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        home_matches = df[df["home_team"] == team_name]
        away_matches = df[df["away_team"] == team_name]

        # Helper for extracting rolling stats
        def extract_stats(matches, prefix, is_home=True):
            if len(matches) == 0:
                return {
                    "avg_gf": 1.5,
                    "avg_ga": 1.5,
                    "xg": 1.5,
                    "elo": 1500,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "form_score": 0,
                }
            recent = matches.tail(num_matches)
            
            # Count wins, draws, losses using result encoding: H=2, D=0, A=1
            if is_home:
                # Home team: wins=2, draws=0, losses=1
                wins = (recent["result"] == 2).sum()
                draws = (recent["result"] == 0).sum()
                losses = (recent["result"] == 1).sum()
            else:
                # Away team: wins=1, draws=0, losses=2
                wins = (recent["result"] == 1).sum()
                draws = (recent["result"] == 0).sum()
                losses = (recent["result"] == 2).sum()
            
            # Calculate form score (3 pts per win, 1 pt per draw)
            form_score = (wins * 3) + draws
            
            return {
                "avg_gf": recent[f"{prefix}_goals_scored_rolling"].mean(),
                "avg_ga": recent[f"{prefix}_goals_conceded_rolling"].mean(),
                "xg": recent["xG"].mean() if prefix == "home" else recent["xG.1"].mean(),
                "elo": recent[f"{prefix}_elo_before"].iloc[-1],
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "form_score": form_score,
            }

        home_stats = extract_stats(home_matches, "home", is_home=True)
        away_stats = extract_stats(away_matches, "away", is_home=False)

        return {"home": home_stats, "away": away_stats}
    
    def load_future_matches_data(self) -> Optional[pd.DataFrame]:
        """Load the future matches data (e.g., 2025-26 season)"""
        try:
            file_path = self.data_dir / "pl25-26.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df
            else:
                return None
        except Exception as e:
            print(f"Error loading future matches data: {e}")
            return None
    
    def get_unseen_matches(self, limit: int = 10) -> list:
        """
        Get list of unseen match tuples (home_team, away_team) from future matches.
        These are matches not in training data.
        """
        df = self.load_future_matches_data()
        if df is None or 'HomeTeam' not in df.columns:
            return []
        
        matches = []
        for _, row in df.iterrows():
            home = row.get('HomeTeam', '')
            away = row.get('AwayTeam', '')
            if home and away:
                matches.append((home, away))
        
        # Return the last 'limit' matches (most recent/unseen)
        return matches[-limit:] if limit else matches
        df = self.load_matches_data()
        if df is None:
            return []

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        results = []

        for _, m in df[df["home_team"] == team_name].iterrows():
            r = m["result"]
            results.append(("W" if r == 2 else "D" if r == 1 else "L", m["date"]))

        for _, m in df[df["away_team"] == team_name].iterrows():
            r = m["result"]
            results.append(("W" if r == 0 else "D" if r == 1 else "L", m["date"]))

        results.sort(key=lambda x: x[1])
        return [r[0] for r in results[-n:]]
    
    def prepare_unseen_match_features(self, home_team: str, away_team: str) -> Optional[pd.DataFrame]:
        """
        Prepare features for unseen matches from pl25-26.csv using ONLY the future dataset.
        This matches the notebook's approach: rolling stats are computed from the future matches themselves,
        not from historical data.
        """
        df_future = self.load_future_matches_data()
        if df_future is None:
            return None
        
        # Rename columns to match notebook
        rename_map = {
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
            "HS": "home_shots",
            "AS": "away_shots",
            "HST": "home_sot",
            "AST": "away_sot",
            "HF": "home_fouls",
            "AF": "away_fouls",
            "HC": "home_corners",
            "AC": "away_corners",
            "HY": "home_yellow",
            "AY": "away_yellow",
            "HR": "home_red",
            "AR": "away_red",
            "B365H": "odds_home_win",
            "B365D": "odds_draw",
            "B365A": "odds_away_win"
        }
        df_future = df_future.rename(columns=rename_map)
        
        # Load and merge xG data
        xg_path = self.data_dir / "pl_all_seasonsXG.csv"
        if xg_path.exists():
            df_xg = pd.read_csv(xg_path)
            
            # Apply team name mapping
            team_name_map = {
                "Newcastle Utd": "Newcastle",
                "Manchester Utd": "Man United",
                "Manchester City": "Man City",
                "Leicester City": "Leicester",
                "Leeds United": "Leeds",
                "Ipswich Town": "Ipswich",
                "Luton Town": "Luton",
                "Norwich City": "Norwich",
                "Sheffield Utd": "Sheffield United",
                "Nott'ham Forest": "Nott'm Forest"
            }
            df_xg['home_team'] = df_xg['home_team'].replace(team_name_map)
            df_xg['away_team'] = df_xg['away_team'].replace(team_name_map)
            
            # Convert dates
            df_future['date'] = pd.to_datetime(df_future['date'], dayfirst=True)
            df_xg['date'] = pd.to_datetime(df_xg['date'])
            
            # Merge on date and teams
            df_merged = df_future.merge(df_xg, on=['date', 'home_team', 'away_team'], how='left')
        else:
            df_merged = df_future.copy()
            df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True)
            df_merged['xG'] = 0
            df_merged['xG.1'] = 0
        
        # Sort by date
        df_merged = df_merged.sort_values('date').reset_index(drop=True)
        
        # Compute rolling stats from the future dataset itself
        def add_rolling_team_stats(df, window=5):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            stat_map = {
                "goals_scored": ("home_goals", "away_goals"),
                "goals_conceded": ("away_goals", "home_goals"),
                "shots": ("home_shots", "away_shots"),
                "shots_on_target": ("home_sot", "away_sot"),
                "fouls": ("home_fouls", "away_fouls"),
                "corners": ("home_corners", "away_corners"),
                "yellow_cards": ("home_yellow", "away_yellow"),
                "red_cards": ("home_red", "away_red"),
                "xG": ("xG", "xG.1"),
                "xG.1": ("xG.1", "xG"),
            }

            # Build long format
            home_df = df.rename(columns={v[0]: k for k, v in stat_map.items()})
            away_df = df.rename(columns={v[1]: k for k, v in stat_map.items()})

            home_df["team"] = home_df["home_team"]
            away_df["team"] = away_df["away_team"]

            cols = ["date", "team"] + list(stat_map.keys())
            long_df = pd.concat([home_df[cols], away_df[cols]], ignore_index=True)
            long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

            # Compute rolling stats (shift by 1 to exclude current match)
            for stat in stat_map.keys():
                long_df[f"{stat}_rolling_avg"] = (
                    long_df.groupby("team")[stat]
                           .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

            # HOME merge
            home_merge_cols = {f"{stat}_rolling_avg": f"home_{stat}_rolling"
                               for stat in stat_map.keys()}

            df = df.merge(
                long_df[["date", "team"] + list(home_merge_cols.keys())],
                left_on=["date", "home_team"],
                right_on=["date", "team"],
                how="left"
            ).drop(columns=["team"])
            df = df.rename(columns=home_merge_cols)

            # AWAY merge
            away_merge_cols = {f"{stat}_rolling_avg": f"away_{stat}_rolling"
                               for stat in stat_map.keys()}

            df = df.merge(
                long_df[["date", "team"] + list(away_merge_cols.keys())],
                left_on=["date", "away_team"],
                right_on=["date", "team"],
                how="left"
            ).drop(columns=["team"])
            df = df.rename(columns=away_merge_cols)

            # Fill NAN
            df = df.fillna(0)

            return df
        
        match_stats = add_rolling_team_stats(df_merged, window=5)
        
        # Now add ELO calculation
        def update_elo(elo_a, elo_b, score_a, score_b, k=20):
            expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
            expected_b = 1 - expected_a

            if score_a > score_b:
                actual_a, actual_b = 1, 0
            elif score_a < score_b:
                actual_a, actual_b = 0, 1
            else:
                actual_a, actual_b = 0.5, 0.5

            new_a = elo_a + k * (actual_a - expected_a)
            new_b = elo_b + k * (actual_b - expected_b)

            return new_a, new_b
        
        def add_elo_features(df):
            df = df.sort_values("date").copy()
            team_elos = {}
            base_elo = 1500

            df["home_elo_before"] = 0.0
            df["away_elo_before"] = 0.0

            for i, row in df.iterrows():
                home, away = row["home_team"], row["away_team"]

                # If new team, initialize
                team_elos.setdefault(home, base_elo)
                team_elos.setdefault(away, base_elo)

                # Assign ELO before match
                df.at[i, "home_elo_before"] = team_elos[home]
                df.at[i, "away_elo_before"] = team_elos[away]

                # Update after match
                new_home, new_away = update_elo(
                    team_elos[home], team_elos[away],
                    row.get("home_goals", 0), row.get("away_goals", 0)
                )

                team_elos[home] = new_home
                team_elos[away] = new_away

            return df
        
        match_stats = add_elo_features(match_stats)
        
        # Add bookmaker probabilities
        def add_bookmaker_features(df):
            df = df.copy()

            # Convert to implied probabilities
            df["prob_home"] = 1 / df["odds_home_win"]
            df["prob_draw"] = 1 / df["odds_draw"]
            df["prob_away"] = 1 / df["odds_away_win"]

            # Normalize to remove overround
            total = df[["prob_home", "prob_draw", "prob_away"]].sum(axis=1)

            df["prob_home"] /= total
            df["prob_draw"] /= total
            df["prob_away"] /= total

            return df
        
        match_stats = add_bookmaker_features(match_stats)
        
        # Add derived features
        match_stats['home_xG_rolling'] = (
            match_stats.groupby('home_team')['xG'].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

        match_stats['away_xG_rolling'] = (
            match_stats.groupby('away_team')['xG.1'].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

        match_stats['xG_diff_rolling'] = match_stats['home_xG_rolling'] - match_stats['away_xG_rolling']
        match_stats['elo_diff'] = match_stats['home_elo_before'] - match_stats['away_elo_before']
        match_stats['goals_diff_rolling'] = match_stats['home_goals_scored_rolling'] - match_stats['away_goals_scored_rolling']
        match_stats['conceded_diff_rolling'] = match_stats['home_goals_conceded_rolling'] - match_stats['away_goals_conceded_rolling']
        
        # Find the specific match
        target_match = match_stats[(match_stats['home_team'] == home_team) & (match_stats['away_team'] == away_team)]
        
        if target_match.empty:
            return None
        
        # Extract features for the model
        row = target_match.iloc[0]
        features = {
            "odds_home_win": row.get("odds_home_win", 2.0),
            "odds_draw": row.get("odds_draw", 3.0),
            "odds_away_win": row.get("odds_away_win", 3.5),
            "home_goals_scored_rolling": row.get("home_goals_scored_rolling", 0),
            "home_goals_conceded_rolling": row.get("home_goals_conceded_rolling", 0),
            "home_shots_rolling": row.get("home_shots_rolling", 0),
            "home_shots_on_target_rolling": row.get("home_shots_on_target_rolling", 0),
            "home_fouls_rolling": row.get("home_fouls_rolling", 0),
            "home_corners_rolling": row.get("home_corners_rolling", 0),
            "home_yellow_cards_rolling": row.get("home_yellow_cards_rolling", 0),
            "away_goals_scored_rolling": row.get("away_goals_scored_rolling", 0),
            "away_goals_conceded_rolling": row.get("away_goals_conceded_rolling", 0),
            "away_shots_rolling": row.get("away_shots_rolling", 0),
            "away_shots_on_target_rolling": row.get("away_shots_on_target_rolling", 0),
            "away_fouls_rolling": row.get("away_fouls_rolling", 0),
            "away_corners_rolling": row.get("away_corners_rolling", 0),
            "away_yellow_cards_rolling": row.get("away_yellow_cards_rolling", 0),
            "home_elo_before": row.get("home_elo_before", 1500),
            "away_elo_before": row.get("away_elo_before", 1500),
            "prob_home": row.get("prob_home", 0.333),
            "prob_draw": row.get("prob_draw", 0.333),
            "prob_away": row.get("prob_away", 0.334),
            "home_xG_rolling": row.get("home_xG_rolling", 0),
            "away_xG_rolling": row.get("away_xG_rolling", 0),
            "xG_diff_rolling": row.get("xG_diff_rolling", 0),
            "elo_diff": row.get("elo_diff", 0),
            "goals_diff_rolling": row.get("goals_diff_rolling", 0),
            "conceded_diff_rolling": row.get("conceded_diff_rolling", 0),
        }
        
        return pd.DataFrame([features])
        """
        Creates a single-row DataFrame with features for model prediction.
        Matches the format from predictmatch.ipynb
        """
        df = self.load_matches_data()
        if df is None:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Get last match for each team
        home_last = df[df["home_team"] == home_team].tail(1)
        away_last = df[df["away_team"] == away_team].tail(1)

        # If either team has no data, return None
        if home_last.empty or away_last.empty:
            return None

        # Get home team's last match (whether home or away)
        home_matches = pd.concat([
            df[df["home_team"] == home_team],
            df[df["away_team"] == home_team]
        ]).sort_values("date").tail(1)

        # Get away team's last match (whether home or away)
        away_matches = pd.concat([
            df[df["home_team"] == away_team],
            df[df["away_team"] == away_team]
        ]).sort_values("date").tail(1)

        # Extract features from most recent games
        if not home_matches.empty:
            home_row = home_matches.iloc[0]
            home_gf_rolling = home_row.get("home_goals_scored_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_goals_scored_rolling", 0)
            home_ga_rolling = home_row.get("home_goals_conceded_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_goals_conceded_rolling", 0)
            home_shots = home_row.get("home_shots_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_shots_rolling", 0)
            home_sot = home_row.get("home_shots_on_target_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_shots_on_target_rolling", 0)
            home_fouls = home_row.get("home_fouls_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_fouls_rolling", 0)
            home_corners = home_row.get("home_corners_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_corners_rolling", 0)
            home_yellow = home_row.get("home_yellow_cards_rolling", 0) if home_row["home_team"] == home_team else home_row.get("away_yellow_cards_rolling", 0)
            home_elo = home_row.get("home_elo_before", 1500) if home_row["home_team"] == home_team else home_row.get("away_elo_before", 1500)
            home_xg = home_row.get("xG", 0) if home_row["home_team"] == home_team else home_row.get("xG.1", 0)
        else:
            home_gf_rolling = home_ga_rolling = home_shots = home_sot = home_fouls = home_corners = home_yellow = 0
            home_elo = 1500
            home_xg = 0

        if not away_matches.empty:
            away_row = away_matches.iloc[0]
            away_gf_rolling = away_row.get("away_goals_scored_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_goals_scored_rolling", 0)
            away_ga_rolling = away_row.get("away_goals_conceded_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_goals_conceded_rolling", 0)
            away_shots = away_row.get("away_shots_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_shots_rolling", 0)
            away_sot = away_row.get("away_shots_on_target_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_shots_on_target_rolling", 0)
            away_fouls = away_row.get("away_fouls_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_fouls_rolling", 0)
            away_corners = away_row.get("away_corners_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_corners_rolling", 0)
            away_yellow = away_row.get("away_yellow_cards_rolling", 0) if away_row["away_team"] == away_team else away_row.get("home_yellow_cards_rolling", 0)
            away_elo = away_row.get("away_elo_before", 1500) if away_row["away_team"] == away_team else away_row.get("home_elo_before", 1500)
            away_xg = away_row.get("xG.1", 0) if away_row["away_team"] == away_team else away_row.get("xG", 0)
        else:
            away_gf_rolling = away_ga_rolling = away_shots = away_sot = away_fouls = away_corners = away_yellow = 0
            away_elo = 1500
            away_xg = 0

        # Compute rolling xG for home and away
        home_xg_rolling = df[df["home_team"] == home_team]["xG"].tail(5).mean() if not df[df["home_team"] == home_team].empty else home_xg
        away_xg_rolling = df[df["away_team"] == away_team]["xG.1"].tail(5).mean() if not df[df["away_team"] == away_team].empty else away_xg

        # Compute differences
        xg_diff_rolling = home_xg_rolling - away_xg_rolling
        elo_diff = home_elo - away_elo
        goals_diff_rolling = home_gf_rolling - away_gf_rolling
        conceded_diff_rolling = home_ga_rolling - away_ga_rolling

        # Get bookmaker probabilities (use recent odds or defaults)
        if not home_last.empty:
            odds_home = home_last.iloc[0].get("odds_home_win", 2.0)
            odds_draw = home_last.iloc[0].get("odds_draw", 3.0)
            odds_away = home_last.iloc[0].get("odds_away_win", 3.5)
            prob_home = home_last.iloc[0].get("prob_home", 0.33)
            prob_draw = home_last.iloc[0].get("prob_draw", 0.33)
            prob_away = home_last.iloc[0].get("prob_away", 0.34)
        else:
            odds_home = odds_draw = odds_away = 2.0
            prob_home = prob_draw = prob_away = 0.333

        features = {
            "odds_home_win": odds_home,
            "odds_draw": odds_draw,
            "odds_away_win": odds_away,
            "home_goals_scored_rolling": home_gf_rolling,
            "home_goals_conceded_rolling": home_ga_rolling,
            "home_shots_rolling": home_shots,
            "home_shots_on_target_rolling": home_sot,
            "home_fouls_rolling": home_fouls,
            "home_corners_rolling": home_corners,
            "home_yellow_cards_rolling": home_yellow,
            "away_goals_scored_rolling": away_gf_rolling,
            "away_goals_conceded_rolling": away_ga_rolling,
            "away_shots_rolling": away_shots,
            "away_shots_on_target_rolling": away_sot,
            "away_fouls_rolling": away_fouls,
            "away_corners_rolling": away_corners,
            "away_yellow_cards_rolling": away_yellow,
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "home_xG_rolling": home_xg_rolling,
            "away_xG_rolling": away_xg_rolling,
            "xG_diff_rolling": xg_diff_rolling,
            "elo_diff": elo_diff,
            "goals_diff_rolling": goals_diff_rolling,
            "conceded_diff_rolling": conceded_diff_rolling,
        }

        return pd.DataFrame([features])
