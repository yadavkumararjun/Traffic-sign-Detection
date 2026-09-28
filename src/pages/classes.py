from pathlib import Path
import sys

import streamlit as st


SRC_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent
)

sys.path.insert(
    0,
    str(SRC_DIR)
)

from predictor import (
    load_class_names
)


@st.cache_data
def load_classes():

    return load_class_names()


def render_classes():

    st.title(
        "Traffic Sign Classes"
    )

    st.write(
        "All 52 traffic-sign classes supported by the model."
    )

    st.divider()

    classes = load_classes()

    search = st.text_input(
        "Search classes",
        placeholder="Search for turn, speed, parking..."
    )

    rows = []

    for class_id, name in classes.items():

        if (
            not search
            or search.lower()
            in name.lower()
        ):

            rows.append(
                {
                    "Class ID": class_id,
                    "Traffic Sign": name
                }
            )

    st.caption(
        f"{len(rows)} class(es)"
    )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        height=600
    )