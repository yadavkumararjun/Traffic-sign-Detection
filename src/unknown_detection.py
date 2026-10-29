import numpy as np


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.70

MARGIN_THRESHOLD = 0.15

ENTROPY_THRESHOLD = 2.5


# ============================================================
# CHECK PREDICTION
# ============================================================

def check_prediction_confidence(
    predictions
):

    probabilities = np.asarray(
        predictions,
        dtype=np.float64
    )

    # Handle (1, 52)
    if probabilities.ndim == 2:

        probabilities = (
            probabilities[0]
        )

    # Safety normalization
    total = probabilities.sum()

    if total > 0:

        probabilities = (
            probabilities / total
        )

    # --------------------------------------------------------
    # Sort predictions
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top_class = int(
        sorted_indices[0]
    )

    second_class = int(
        sorted_indices[1]
    )

    top_confidence = float(
        probabilities[top_class]
    )

    second_confidence = float(
        probabilities[second_class]
    )

    margin = (
        top_confidence
        - second_confidence
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    safe_probabilities = np.clip(
        probabilities,
        1e-10,
        1.0
    )

    entropy = float(
        -np.sum(
            safe_probabilities
            * np.log(
                safe_probabilities
            )
        )
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if (
        top_confidence
        < CONFIDENCE_THRESHOLD
    ):

        status = "unknown"

    elif (
        margin
        < MARGIN_THRESHOLD
    ):

        status = "uncertain"

    elif (
        entropy
        > ENTROPY_THRESHOLD
    ):

        status = "uncertain"

    else:

        status = "recognized"

    return {

        "top_class": top_class,

        "second_class": second_class,

        "top_confidence":
            top_confidence,

        "second_confidence":
            second_confidence,

        "margin":
            margin,

        "entropy":
            entropy,

        "status":
            status,

        "probabilities":
            probabilities
    }


# ============================================================
# GET STATUS
# ============================================================

def get_prediction_status(
    result
):

    return result["status"]