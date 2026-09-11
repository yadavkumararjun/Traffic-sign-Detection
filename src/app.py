import streamlit as st

from components.sidebar import show_sidebar

from pages.dashboard import show_dashboard
from pages.detect import show_detect
from pages.model_info import show_model_info
from pages.traffic_signs import show_traffic_signs


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SignVision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = show_sidebar()


# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------

if page == "🏠 Dashboard":

    show_dashboard()


elif page == "🖼️ Detect Sign":

    show_detect()


elif page == "📊 Model":

    show_model_info()


elif page == "📚 Traffic Signs":

    show_traffic_signs()