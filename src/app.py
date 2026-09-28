from pathlib import Path
import sys

import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Sign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# IMPORT PAGES
# ============================================================

from dashboard import (
    render_sidebar,
    render_dashboard
)

from pages.prediction import (
    render_prediction
)

from pages.performance import (
    render_performance
)

from pages.classes import (
    render_classes
)

from pages.about import (
    render_about
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

selected_page = render_sidebar()


# ============================================================
# ROUTING
# ============================================================

if selected_page == "Dashboard":

    render_dashboard()


elif selected_page == "Prediction":

    render_prediction()


elif selected_page == "Performance":

    render_performance()


elif selected_page == "Traffic Classes":

    render_classes()


elif selected_page == "About":

    render_about()


else:

    render_dashboard()