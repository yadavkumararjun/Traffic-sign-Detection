import streamlit as st


def show_model_info():

    st.title("📊 Model Information")

    st.write(
        "Details about the trained Indian Traffic Sign "
        "Classification model."
    )

    st.divider()

    # --------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------

    st.subheader("🤖 Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Test Accuracy",
            "89.95%"
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
            "CNN"
        )

    st.divider()

    # --------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("📋 Model Details")

        st.write("**Model Type:** Convolutional Neural Network")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Input Shape:** 32 × 32 × 3")
        st.write("**Output Classes:** 52")
        st.write("**Task:** Image Classification")
        st.write("**Dataset:** Indian Traffic Sign Dataset")

    with right_col:

        st.subheader("⚙️ Preprocessing")

        st.write("**Color Format:** RGB")
        st.write("**Image Resize:** 32 × 32")
        st.write("**Pixel Scaling:** 0 – 1")
        st.write("**Normalization:** Pixel / 255")
        st.write("**Data Split:** 80% Training / 20% Testing")

    st.divider()

    # --------------------------------------------------
    # CNN ARCHITECTURE
    # --------------------------------------------------

    st.subheader("🧠 CNN Architecture")

    st.code(
        """
Input Image
32 × 32 × 3
      ↓
Conv2D
32 Filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D
64 Filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Conv2D
128 Filters
      ↓
Batch Normalization
      ↓
Max Pooling
      ↓
Flatten
      ↓
Dense
128 Neurons
      ↓
Dropout
0.5
      ↓
Output Layer
52 Classes
        """,
        language="text"
    )

    st.divider()

    # --------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------

    st.subheader("📈 Model Performance")

    performance_col1, performance_col2 = st.columns(2)

    with performance_col1:

        st.metric(
            "Test Accuracy",
            "89.95%"
        )

        st.progress(0.8995)

    with performance_col2:

        st.write("**Evaluation Dataset**")
        st.write("2,795 test images")

        st.write("**Training Dataset**")
        st.write("11,176 training images")

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------

    st.subheader("🛠️ Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.info(
            "**Python**\n\n"
            "Programming Language"
        )

        st.info(
            "**TensorFlow / Keras**\n\n"
            "Deep Learning"
        )

    with tech2:

        st.info(
            "**OpenCV**\n\n"
            "Image Processing"
        )

        st.info(
            "**NumPy**\n\n"
            "Numerical Computing"
        )

    with tech3:

        st.info(
            "**Streamlit**\n\n"
            "Web Application"
        )

        st.info(
            "**scikit-learn**\n\n"
            "Evaluation & Analysis"
        )

    st.divider()

    # --------------------------------------------------
    # MODEL FILE
    # --------------------------------------------------

    st.subheader("📁 Model File")

    st.write(
        "The application uses the following trained model:"
    )

    st.code(
        "models/traffic_sign_targeted_model.keras",
        language="text"
    )

    st.success(
        "✅ Trained model loaded successfully by the "
        "prediction pipeline."
    )