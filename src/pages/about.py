import streamlit as st


def render_about():

    st.title(
        "ℹ️ About"
    )

    st.write(
        """
        Traffic Sign AI is a deep-learning based traffic sign
        recognition system.
        """
    )

    st.divider()

    st.subheader(
        "Technology"
    )

    st.write(
        """
        • Python
        • TensorFlow
        • Keras
        • OpenCV
        • NumPy
        • Streamlit
        """
    )

    st.divider()

    st.subheader(
        "Model"
    )

    st.write(
        """
        The application uses a robust CNN model trained to
        recognize 52 traffic-sign classes.
        """
    )

    st.metric(
        "Test Accuracy",
        "94.20%"
    )

    st.metric(
        "Number of Classes",
        "52"
    )

    st.metric(
        "Test Images",
        "2,795"
    )

    st.divider()

    st.info(
        "Camera-based recognition is intentionally not included yet."
    )