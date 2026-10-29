import streamlit as st


def render_performance():

    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Performance evaluation of the robust CNN model."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            "94.20%"
        )

    with col2:

        st.metric(
            "Macro F1",
            "93.70%"
        )

    with col3:

        st.metric(
            "Weighted F1",
            "94.30%"
        )

    st.divider()

    st.subheader(
        "Model Information"
    )

    st.write(
        """
        The robust model was trained using targeted
        augmentation and evaluated on 2,795 test images.
        """
    )

    st.write(
        """
        The model contains 52 semantic traffic-sign classes.
        """
    )

    st.success(
        "Robust CNN model achieved 94.20% test accuracy."
    )