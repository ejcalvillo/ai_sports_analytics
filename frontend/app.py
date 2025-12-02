"""
AI Sports Analytics Dashboard
Main Streamlit Application
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import from project root
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Page configuration
st.set_page_config(
    page_title="AI Sports Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚽ AI Sports Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🔮 Match Predictor", "📊 Model Performance"]
)



# Import page modules
try:
    if page == "🔮 Match Predictor":
        from pages import predictor
        predictor.show()
    elif page == "📊 Model Performance":
        from pages import performance
        performance.show()
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 1rem;'>"
    "AI Sports Analytics | Premier League Prediction System"
    "</div>",
    unsafe_allow_html=True
)
