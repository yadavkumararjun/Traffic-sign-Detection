import streamlit as st


def render_about():

    st.title(
        "About the Project"
    )

    st.write(
        "CNN-based Traffic Sign Recognition System"
    )

    st.divider()

    st.subheader(
        "Project Overview"
    )

    st.write(
        "This project uses a Convolutional Neural Network "
        "to classify traffic-sign images into 52 semantic "
        "traffic-sign classes."
    )

    st.subheader(
        "Final Model"
    )

    st.write(
        """
        - Architecture: Robust CNN
        - Input: 32 × 32 RGB
        - Classes: 52
        - Test accuracy: 94.20%
        - Test loss: 0.1904
        """
    )

    st.subheader(
        "Image Processing"
    )

    st.write(
        "Uploaded images are converted to RGB, resized while "
        "preserving their aspect ratio, centered inside a "
        "32 × 32 canvas, normalized, and passed to the CNN."
    )

    st.info(
        "Camera functionality is intentionally disabled "
        "in this version."
    )