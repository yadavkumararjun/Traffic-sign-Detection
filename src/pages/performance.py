import streamlit as st


def render_performance():

    st.title(
        "Model Performance"
    )

    st.write(
        "Evaluation results from the robust CNN model."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Test Accuracy",
            "94.20%"
        )

    with col2:

        st.metric(
            "Test Loss",
            "0.1904"
        )

    with col3:

        st.metric(
            "Classes",
            "52"
        )

    st.subheader(
        "Important Class Performance"
    )

    st.dataframe(
        [
            {
                "Class": "Turn left",
                "Accuracy": "52.50%"
            },
            {
                "Class": "Turn right",
                "Accuracy": "47.50%"
            },
            {
                "Class": "Steep ascent",
                "Accuracy": "77.50%"
            },
            {
                "Class": "Steep descent",
                "Accuracy": "72.50%"
            },
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Major Confusion Pairs"
    )

    st.dataframe(
        [
            {
                "True Class": "Turn left",
                "Predicted As": "Turn right",
                "Mistakes": 18
            },
            {
                "True Class": "Turn right",
                "Predicted As": "Turn left",
                "Mistakes": 17
            },
            {
                "True Class": "Steep descent",
                "Predicted As": "Steep ascent",
                "Mistakes": 11
            },
            {
                "True Class": "No stopping",
                "Predicted As": "Horn prohibited",
                "Mistakes": 9
            },
            {
                "True Class": "Steep ascent",
                "Predicted As": "Steep descent",
                "Mistakes": 7
            },
        ],
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The remaining difficult cases are concentrated "
        "around visually similar signs, especially directional "
        "and road-gradient signs."
    )