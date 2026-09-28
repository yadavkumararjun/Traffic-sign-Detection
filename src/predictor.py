from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

CLASS_PATH = (
    ROOT
    / "data"
    / "semantic_class_names.txt"
)


def load_class_names():

    classes = {}

    with open(
        CLASS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            class_id, name = line.split(
                ",",
                1
            )

            classes[int(class_id)] = name

    return classes


def preprocess_image(
    image,
    image_size=32
):

    # PIL → NumPy
    image = np.array(
        image.convert("RGB")
    )

    # RGB → BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    height, width = image.shape[:2]

    # Preserve aspect ratio
    scale = min(
        image_size / width,
        image_size / height
    )

    new_width = max(
        1,
        round(width * scale)
    )

    new_height = max(
        1,
        round(height * scale)
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    # 32 × 32 canvas
    canvas = np.zeros(
        (
            image_size,
            image_size,
            3
        ),
        dtype=np.uint8
    )

    x = (
        image_size - new_width
    ) // 2

    y = (
        image_size - new_height
    ) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    # BGR → RGB
    canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    canvas = (
        canvas.astype(
            np.float32
        ) / 255.0
    )

    # Batch dimension
    canvas = np.expand_dims(
        canvas,
        axis=0
    )

    return canvas


def predict(
    model,
    image,
    class_names
):

    processed = preprocess_image(
        image
    )

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    results = []

    for index in top_indices:

        index = int(index)

        results.append(
            {
                "id": index,
                "name": class_names.get(
                    index,
                    f"Class {index}"
                ),
                "confidence":
                    float(
                        probabilities[index]
                    ) * 100
            }
        )

    return results