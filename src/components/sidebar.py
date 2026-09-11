import streamlit as st


def show_sidebar():

    with st.sidebar:

        st.title("🚦 SignVision")

        st.caption(
            "Indian Traffic Sign Detection"
        )

        st.divider()

        st.subheader("Navigation")

        page = st.radio(
            "Go to",
            [
                "🏠 Dashboard",
                "🖼️ Detect Sign",
                "📊 Model",
                "📚 Traffic Signs"
            ]
        )

        st.divider()

        st.success("● Model Online")

        st.caption("CNN Classification System")
        st.caption("52 Semantic Classes")

    return page