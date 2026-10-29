import streamlit as st


def render_classes():

    st.title(
        "🚦 Traffic Sign Classes"
    )

    st.write(
        "The model recognizes 52 traffic-sign classes."
    )

    st.divider()

    classes = [

        "Axle weight limit",
        "Bus stop",
        "Cattle",
        "Crossroads",
        "Dip",
        "Falling rocks",
        "Filling station",
        "First aid post",
        "Give way",
        "Guarded level crossing ahead",
        "Height limit",
        "Horn prohibited",
        "Hotel",
        "Length limit",
        "Level crossing countdown marker",
        "Loose gravel",
        "Maximum speed limit (110 km/h)",
        "Maximum speed limit (90 km/h)",
        "Narrow bridge",
        "Narrow road",
        "No entry",
        "No entry for bullock carts",
        "No entry for cycles",
        "No entry for goods vehicles",
        "No entry for hand carts",
        "No entry for motor vehicles",
        "No entry for pedestrians",
        "No left turn",
        "No overtaking",
        "No parking",
        "No right turn",
        "No stopping",
        "No vehicles in both directions",
        "Oblique side road junction",
        "One-way traffic",
        "Parking",
        "Refreshments",
        "Restaurant",
        "Road hump",
        "Roundabout",
        "Side road junction",
        "Staggered side road junction",
        "Steep ascent",
        "Steep descent",
        "T-junction",
        "Telephone",
        "Turn left",
        "Turn right",
        "Unguarded level crossing ahead",
        "Unprotected quay",
        "Weight limit",
        "Y-junction"
    ]

    for i in range(
        0,
        len(classes),
        2
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**{i}.** {classes[i]}"
            )

        if i + 1 < len(classes):

            with col2:

                st.write(
                    f"**{i + 1}.** {classes[i + 1]}"
                )