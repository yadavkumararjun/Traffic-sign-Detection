import streamlit as st
import numpy as np
from PIL import Image

from predictor import (
    load_model,
    predict_image,
    load_class_names
)

from unknown_detection import (
    check_prediction_confidence
)


# ============================================================
# CACHE MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_model():

    return load_model()


# ============================================================
# CACHE CLASS NAMES
# ============================================================

@st.cache_data
def get_classes():

    return load_class_names()


# ============================================================
# PREDICTION PAGE
# ============================================================

def render_prediction():

    st.title(
        "🚦 Traffic Sign Prediction"
    )

    st.write(
        "Upload an image and let the CNN identify "
        "the traffic sign."
    )

    st.divider()

    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(

        "Upload traffic sign image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload a traffic sign image to begin."
        )

        return

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # ========================================================
    # DISPLAY
    # ========================================================

    image_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    with image_col:

        st.subheader(
            "Input Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    with result_col:

        st.subheader(
            "Prediction"
        )

        predict_button = st.button(

            "🔍 Predict Traffic Sign",

            type="primary",

            use_container_width=True
        )

        if not predict_button:

            st.info(
                "Click the button to analyze the image."
            )

            return

        # ====================================================
        # MODEL
        # ====================================================

        with st.spinner(
            "Analyzing image..."
        ):

            model = get_model()

            class_names = get_classes()

            predictions = predict_image(
                model,
                image
            )

        # ====================================================
        # UNKNOWN DETECTION
        # ====================================================

        result = (
            check_prediction_confidence(
                predictions
            )
        )

        status = result["status"]

        top_class = result[
            "top_class"
        ]

        confidence = result[
            "top_confidence"
        ]

        # ====================================================
        # UNKNOWN
        # ====================================================

        if status == "unknown":

            st.error(
                "❌ Unknown / Unrecognized Image"
            )

            st.write(
                "The model is not confident enough "
                "that this image belongs to one of "
                "the supported traffic-sign classes."
            )

            st.metric(
                "Highest confidence",
                f"{confidence * 100:.2f}%"
            )

            st.metric(
                "Prediction margin",
                f"{result['margin'] * 100:.2f}%"
            )

            return

        # ====================================================
        # UNCERTAIN
        # ====================================================

        if status == "uncertain":

            st.warning(
                "⚠️ Prediction is uncertain"
            )

            st.caption(
                "The model sees more than one possible class."
            )

        # ====================================================
        # RECOGNIZED
        # ====================================================

        predicted_name = class_names[
            top_class
        ]

        st.success(
            f"🚦 {predicted_name}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # ====================================================
        # DETAILS
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Details"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Top confidence",
                f"{confidence * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Prediction margin",
                f"{result['margin'] * 100:.2f}%"
            )

        # ====================================================
        # TOP 5
        # ====================================================

        st.subheader(
            "Top 5 Predictions"
        )

        probabilities = result[
            "probabilities"
        ]

        top_indices = np.argsort(
            probabilities
        )[::-1][:5]

        for index in top_indices:

            probability = float(
                probabilities[index]
            )

            name = class_names[
                index
            ]

            st.write(
                f"**{name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.caption(
            "Model: Robust CNN"
        )

        st.caption(
            "Classes: 52"
        )

        st.caption(
            "Test accuracy: 94.20%"
        )