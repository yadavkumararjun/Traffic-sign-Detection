import streamlit as st


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
        margin-bottom: 30px;
    }

    .stat-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMPORT PAGES
# ============================================================

from pages.prediction import render_prediction
from pages.performance import render_performance
from pages.classes import render_classes
from pages.about import render_about


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🚦 Traffic Sign AI"
    )

    st.caption(
        "CNN-based Traffic Sign Recognition"
    )

    st.divider()

    page = st.radio(

        "Navigation",

        [
            "🏠 Dashboard",
            "🔍 Predict Sign",
            "📊 Model Performance",
            "🚦 Traffic Classes",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption(
        "Robust CNN Model"
    )

    st.caption(
        "52 Traffic Sign Classes"
    )

    st.caption(
        "94.20% Test Accuracy"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🚦 Traffic Sign AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent traffic sign recognition using a trained CNN model.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Traffic Classes",
            "52"
        )

    with col2:

        st.metric(
            "Test Accuracy",
            "94.20%"
        )

    with col3:

        st.metric(
            "Test Images",
            "2,795"
        )

    with col4:

        st.metric(
            "Model",
            "CNN"
        )

    st.divider()

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    st.subheader(
        "What does this application do?"
    )

    st.write(
        """
        This application uses a Convolutional Neural Network
        trained on Indian traffic sign images to recognize
        traffic signs from uploaded images.
        """
    )

    st.write(
        """
        The system supports 52 traffic-sign classes and
        includes confidence-based detection to identify
        predictions that may be unreliable.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "Features"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### 🔍 Prediction"
        )

        st.write(
            "Upload an image and identify the traffic sign."
        )

    with col2:

        st.markdown(
            "### 📊 Performance"
        )

        st.write(
            "View model accuracy and classification results."
        )

    with col3:

        st.markdown(
            "### 🚦 Classes"
        )

        st.write(
            "Explore all 52 supported traffic signs."
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔍 Predict Sign":

    render_prediction()


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    render_performance()


# ============================================================
# CLASSES
# ============================================================

elif page == "🚦 Traffic Classes":

    render_classes()


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    render_about()