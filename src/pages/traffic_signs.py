import streamlit as st


CLASS_NAMES_PATH = "data/semantic_class_names.txt"


def load_class_names():

    class_names = []

    try:

        with open(CLASS_NAMES_PATH, "r") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                parts = line.split(",", 1)

                if len(parts) == 2:

                    class_id = int(parts[0])
                    name = parts[1]

                    class_names.append(
                        (class_id, name)
                    )

    except FileNotFoundError:

        st.error(
            "❌ semantic_class_names.txt was not found."
        )

    return class_names


def show_traffic_signs():

    st.title("📚 Traffic Sign Classes")

    st.write(
        "The model recognizes 52 semantic traffic sign "
        "categories."
    )

    st.divider()

    # --------------------------------------------------
    # LOAD CLASSES
    # --------------------------------------------------

    class_names = load_class_names()

    if not class_names:

        st.warning(
            "No traffic sign classes were found."
        )

        return

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    search = st.text_input(
        "🔎 Search Traffic Signs",
        placeholder="Example: parking, turn, speed, entry..."
    )

    # --------------------------------------------------
    # FILTER
    # --------------------------------------------------

    if search:

        filtered_classes = [
            item
            for item in class_names
            if search.lower() in item[1].lower()
        ]

    else:

        filtered_classes = class_names

    # --------------------------------------------------
    # RESULT COUNT
    # --------------------------------------------------

    st.write(
        f"Showing **{len(filtered_classes)}** "
        f"of **{len(class_names)}** classes."
    )

    st.divider()

    # --------------------------------------------------
    # DISPLAY CLASSES
    # --------------------------------------------------

    for class_id, name in filtered_classes:

        with st.container(border=True):

            col1, col2 = st.columns([1, 6])

            with col1:

                st.write(
                    f"**Class {class_id}**"
                )

            with col2:

                st.write(
                    f"**{name}**"
                )

    # --------------------------------------------------
    # NO SEARCH RESULTS
    # --------------------------------------------------

    if search and not filtered_classes:

        st.warning(
            f"No traffic sign found matching "
            f"**'{search}'**."
        )