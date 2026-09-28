from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "models"
    / "traffic_sign_robust_model.keras"
)

CLASS_PATH = (
    ROOT
    / "data"
    / "semantic_class_names.txt"
)


def render_sidebar():

    with st.sidebar:

        st.title("🚦 Traffic Sign AI")

        st.caption(
            "Traffic Sign Recognition System"
        )

        st.divider()

        pages = [
            "Dashboard",
            "Predict Sign",
            "Model Performance",
            "Traffic Sign Classes",
            "About",
        ]

        current_page = st.session_state.get(
            "page",
            "Dashboard"
        )

        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(current_page),
        )

        st.session_state.page = selected_page

        st.divider()

        st.caption("MODEL")

        st.write("Robust CNN")
        st.write("52 classes")
        st.write("94.20% accuracy")


def render_dashboard():

    st.title("Traffic Sign AI")

    st.write(
        "Traffic sign recognition powered by "
        "your trained CNN model."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Test Accuracy",
            "94.20%"
        )

    with col2:
        st.metric(
            "Classes",
            "52"
        )

    with col3:
        st.metric(
            "Input Size",
            "32 × 32"
        )

    with col4:
        st.metric(
            "Model",
            "Robust CNN"
        )

    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:

        if MODEL_PATH.exists():

            st.success(
                "✓ Model loaded"
            )

        else:

            st.error(
                "Model file not found"
            )

    with col2:

        if CLASS_PATH.exists():

            st.success(
                "✓ Class mapping available"
            )

        else:

            st.error(
                "Class mapping not found"
            )

    st.subheader("Quick Start")

    st.info(
        "Go to **Predict Sign** from the sidebar, "
        "upload an image, and click **Predict Sign**."
    )

    st.subheader("Model Highlights")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Recognition**

            - 52 traffic-sign classes
            - CNN-based classification
            - 32 × 32 RGB input
            - Top-5 predictions
            """
        )

    with col2:

        st.markdown(
            """
            **Inference**

            - Aspect-ratio preserving resize
            - Center padding
            - Pixel normalization
            - No image stretching
            """
        )