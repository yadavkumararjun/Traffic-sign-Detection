import streamlit as st
import numpy as np
import cv2
from PIL import Image

from prediction import predict


def show_detect():

    st.title("🖼️ Traffic Sign Detection")

    st.write(
        "Upload an image of a traffic sign and let the "
        "trained CNN model identify it."
    )

    st.divider()

    # --------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG and JPEG"
    )

    # --------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------

    if uploaded_file is None:

        st.info(
            "👆 Upload a traffic sign image to begin detection."
        )

        st.subheader("Supported Formats")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("🖼️ PNG")

        with col2:
            st.info("📷 JPG")

        with col3:
            st.info("📷 JPEG")

        return

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------

    image = Image.open(uploaded_file)

    # Make sure image has exactly 3 RGB channels
    image = image.convert("RGB")

    # --------------------------------------------------
    # IMAGE PREVIEW + INFORMATION
    # --------------------------------------------------

    image_col, info_col = st.columns([1.3, 1])

    with image_col:

        st.subheader("📷 Input Image")

        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )

    with info_col:

        st.subheader("Image Information")

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Original Size:** "
            f"{image.width} × {image.height}"
        )

        st.write(
            "**Model Input:** 32 × 32 × 3"
        )

        st.write(
            "**Preprocessing:** RGB + Normalization"
        )

        st.divider()

        analyze = st.button(
            "🔍 Analyze Traffic Sign",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------

    if analyze:

        with st.spinner("Analyzing traffic sign..."):

            # PIL Image → NumPy array
            image_array = np.array(image)

            # RGB → BGR
            image_bgr = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            # Model prediction
            results = predict(
                image_bgr,
                top_k=5
            )

        # --------------------------------------------------
        # MAIN RESULT
        # --------------------------------------------------

        best = results[0]

        st.divider()

        st.subheader("🎯 Detection Result")

        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:

            st.success(
                f"Detected Sign: **{best['name']}**"
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{best['confidence']:.2f}%"
            )

        # --------------------------------------------------
        # CONFIDENCE MESSAGE
        # --------------------------------------------------

        if best["confidence"] >= 80:

            st.success(
                "🟢 High confidence prediction"
            )

        elif best["confidence"] >= 50:

            st.warning(
                "🟡 Moderate confidence prediction. "
                "The result may need verification."
            )

        else:

            st.error(
                "🔴 Low confidence prediction. "
                "Try uploading a clearer image."
            )

        # --------------------------------------------------
        # TOP 5 PREDICTIONS
        # --------------------------------------------------

        st.divider()

        st.subheader("🏆 Top 5 Predictions")

        for i, result in enumerate(results):

            rank_col, name_col, confidence_col = st.columns(
                [0.6, 3, 1]
            )

            with rank_col:

                st.write(f"### {i + 1}")

            with name_col:

                st.write(
                    f"**{result['name']}**"
                )

                st.progress(
                    min(
                        result["confidence"] / 100,
                        1.0
                    )
                )

            with confidence_col:

                st.write(
                    f"**{result['confidence']:.2f}%**"
                )

        # --------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------

        st.divider()

        st.subheader("ℹ️ About this prediction")

        st.write(
            "The CNN model compares the uploaded image with "
            "patterns learned during training. The confidence "
            "score represents the model's predicted probability "
            "for each class."
        )

        st.caption(
            "A high confidence score does not guarantee that "
            "the prediction is correct."
        )