import streamlit as st


def show_dashboard():

    st.title("Indian Traffic Sign Detection")

    st.subheader(
        "AI-powered traffic sign recognition system"
    )

    st.write(
        "Upload an image of a traffic sign and let the "
        "trained CNN model identify the most likely sign."
    )

    st.divider()

    # --------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model Accuracy",
            "89.95%"
        )

    with col2:

        st.metric(
            "Semantic Classes",
            "52"
        )

    with col3:

        st.metric(
            "Image Size",
            "32 × 32"
        )

    with col4:

        st.metric(
            "Model Type",
            "CNN"
        )

    st.divider()

    # --------------------------------------------------
    # DETECTION SECTION
    # --------------------------------------------------

    left, right = st.columns([2, 1])

    with left:

        st.subheader("🚀 Start Detection")

        st.write(
            "Upload a traffic sign image to get the "
            "predicted sign and Top-5 confidence scores."
        )

    with right:

        st.info(
            "**Input**\n\n"
            "PNG / JPG / JPEG\n\n"
            "**Output**\n\n"
            "Top-5 Predictions"
        )

    st.divider()

    # --------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------

    st.subheader("How it works")

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown("### 1️⃣ Upload")

        st.write(
            "Upload an image containing an Indian "
            "traffic sign."
        )

    with step2:

        st.markdown("### 2️⃣ Process")

        st.write(
            "The image is resized to 32×32 and "
            "normalized before prediction."
        )

    with step3:

        st.markdown("### 3️⃣ Predict")

        st.write(
            "The CNN predicts the most likely traffic "
            "sign and confidence scores."
        )