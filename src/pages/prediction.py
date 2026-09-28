from pathlib import Path
import sys

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PAGE_DIR.parent
ROOT = SRC_DIR.parent

MODEL_PATH = (
    ROOT
    / "models"
    / "traffic_sign_robust_model.keras"
)

CLASS_NAMES_PATH = (
    ROOT
    / "data"
    / "semantic_class_names.txt"
)


# ============================================================
# UNKNOWN DETECTION
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unknown_detection import (
    check_prediction_confidence,
    get_prediction_status
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data(show_spinner=False)
def load_class_names():

    class_names = []

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                ",",
                1
            )

            if len(parts) == 2:

                class_names.append(
                    parts[1]
                )

    return class_names


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = np.array(image)

    # --------------------------------------------------------
    # Resize to model input size
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        (32, 32)
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image
):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    return predictions


# ============================================================
# PAGE
# ============================================================

def render_prediction():

    st.title(
        "🚦 Traffic Sign Prediction"
    )

    st.write(
        "Upload a traffic-sign image and let the trained CNN model analyze it."
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
        ],
        help="Upload a clear traffic-sign image."
    )


    if uploaded_file is None:

        st.info(
            "Upload an image to begin."
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

            st.caption(
                "Click the button to analyze the uploaded image."
            )

            return


        # ====================================================
        # MODEL
        # ====================================================

        with st.spinner(
            "Analyzing image..."
        ):

            model = load_model()

            class_names = load_class_names()

            predictions = predict_image(
                model,
                image
            )


        # ====================================================
        # UNKNOWN DETECTION
        # ====================================================

        confidence_result = (
            check_prediction_confidence(
                predictions
            )
        )

        status = get_prediction_status(
            confidence_result
        )


        # ====================================================
        # GET VALUES
        # ====================================================

        top_class = confidence_result[
            "top_class"
        ]

        top_confidence = confidence_result[
            "top_confidence"
        ]

        margin = confidence_result[
            "margin"
        ]


        # ====================================================
        # UNKNOWN
        # ====================================================

        if status == "unknown":

            st.error(
                "⚠️ Unknown / Unrecognized Image"
            )

            st.write(
                "The model is not sufficiently confident "
                "that this image belongs to one of the "
                "supported traffic-sign classes."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Highest Confidence",
                    f"{top_confidence * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Prediction Margin",
                    f"{margin * 100:.2f}%"
                )

            st.warning(
                "The prediction was rejected instead of "
                "forcing an unsupported image into a class."
            )

            return


        # ====================================================
        # UNCERTAIN
        # ====================================================

        if status == "uncertain":

            st.warning(
                "⚠️ Prediction is uncertain"
            )

            st.write(
                "The model sees multiple possible classes "
                "with similar confidence."
            )


        # ====================================================
        # PREDICTED CLASS
        # ========================================================

        if (
            top_class >= 0
            and top_class < len(class_names)
        ):

            predicted_name = class_names[
                top_class
            ]

        else:

            predicted_name = (
                f"Class {top_class}"
            )


        st.success(
            f"Predicted Sign: {predicted_name}"
        )

        st.metric(
            "Confidence",
            f"{top_confidence * 100:.2f}%"
        )


        # ====================================================
        # TOP 5 PREDICTIONS
        # ====================================================

        st.divider()

        st.subheader(
            "Top Predictions"
        )

        probability_array = (
            predictions[0]
            if predictions.ndim == 2
            else predictions
        )

        top_indices = np.argsort(
            probability_array
        )[::-1][:5]


        for index in top_indices:

            probability = float(
                probability_array[index]
            )

            if (
                index >= 0
                and index < len(class_names)
            ):

                class_name = class_names[
                    index
                ]

            else:

                class_name = (
                    f"Class {index}"
                )


            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                min(
                    probability,
                    1.0
                )
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.caption(
                "Model: Robust CNN"
            )

        with col2:

            st.caption(
                "Supported classes: 52"
            )