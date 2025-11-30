# AI Sports Analytics Frontend

A streamlined Streamlit dashboard for visualizing Premier League match predictions and model performance using a trained RandomForest AI model.

## 🎯 Overview

This frontend provides:
- **Match Predictor**: Select two teams and get AI-powered predictions from the trained RandomForest model
- **Model Performance**: View accuracy, F1-score, confusion matrix, and prediction analysis

## 🤖 AI Model Integration

The dashboard uses the **trained RandomForest model** from `classification.ipynb`:
- **Model**: RandomForestClassifier with 2000 estimators
- **Features**: 50+ features including ELO ratings, recent form (last 5 matches), xG, goals scored/conceded
- **Performance**: 67% accuracy, 59% F1-score on 24 test matches
- **Location**: `../models/random_forest_model.pkl`

Predictions are made by the actual AI model, not simple heuristics!

## 🔄 How It Works - Dynamic Updates

**IMPORTANT**: This dashboard is designed to automatically reflect changes made OUTSIDE the frontend directory.

### Data Flow:
```
classification.ipynb
    ↓ (trains model and saves)
../models/random_forest_model.pkl
    ↓ (loaded by)
Frontend Predictor
    ↓ (makes predictions)
Live AI Predictions

classification.ipynb
    ↓ (saves test results to)
../data/currmatches.csv
    ↓ (read by)
Performance Page
    ↓ (displays)
Live Metrics & Confusion Matrix
```

### Making Model Updates:
1. **Update and train your model** in `classification.ipynb`
2. **Run the model saving cell** (automatically saves to `../models/random_forest_model.pkl`)
3. **Refresh the dashboard** - new predictions use the updated model automatically!
4. **Performance metrics** update automatically from `currmatches.csv`

No frontend code changes needed - the dashboard automatically loads the latest model and reads data files.

## 📁 Directory Structure

```
frontend/
├── app.py                 # Main dashboard entry point
├── pages/
│   ├── predictor.py      # Team selection & predictions
│   └── performance.py    # Model metrics & confusion matrix
├── utils/
│   └── data_loader.py    # Loads data from ../data/
├── requirements.txt      # Python dependencies
└── start.sh             # One-command launcher
```

## 🚀 Quick Start

### Setup (First Time)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ./start.sh
   ```

This script will:
- Create a virtual environment (if needed)
- Install all dependencies
- Launch the dashboard at http://localhost:8501

### Running After Setup

Just run `./start.sh` again - it will reuse the existing environment.

## 📊 Understanding the Dashboard

### Match Predictor Page
- **Purpose**: Get predictions for any two teams
- **How it works**: Uses ELO ratings from your data
- **Shows**: Win probabilities, expected scores, confidence levels

### Model Performance Page
- **Purpose**: Evaluate your model's accuracy
- **Data source**: Reads from `../data/currmatches.csv`
- **Shows**: 
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (matches your notebook output)
  - Prediction breakdown by outcome
  - Actual vs Predicted comparison

## 🔧 Customization

### To Update Your AI Model:
1. Edit and improve your model in `classification.ipynb`
2. Train the model (run the training cell)
3. Run the model saving cell:
   ```python
   import joblib
   joblib.dump(model, '../models/random_forest_model.pkl')
   joblib.dump(feature_names, '../models/feature_names.pkl')
   ```
4. Dashboard automatically uses the new model on next page refresh!

### Model Requirements:
- Must be a scikit-learn compatible model with `predict()` and `predict_proba()` methods
- Feature names must match the training data
- Result encoding: `0=Draw`, `1=Loss (home team loss)`, `2=Win (home team win)`
- Save test results to `../data/currmatches.csv` for performance metrics

## 📦 Dependencies

- Python 3.13+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.17.0
- scikit-learn >= 1.3.0

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

**Dashboard won't start:**
- Delete `venv/` folder and run `./start.sh` again

**Data not showing:**
- Check that `../data/currmatches.csv` exists
- Verify CSV has required columns

**Old results showing:**
- Hard refresh browser: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
- Or click hamburger menu ☰ → "Clear cache"

## 💡 Tips

- **Refresh the page** after updating your model/data
- **Results update automatically** - no code changes needed
- **Confusion matrix** will match your notebook if data is synced
- **Keep frontend/ directory clean** - all model work stays in parent directory

## 📝 Result Encoding

The dashboard uses the same encoding as your notebook:
- `0` = Draw
- `1` = Loss (home team loss / away team win)
- `2` = Win (home team win)

This ensures the confusion matrix and metrics match your Jupyter notebook exactly.
