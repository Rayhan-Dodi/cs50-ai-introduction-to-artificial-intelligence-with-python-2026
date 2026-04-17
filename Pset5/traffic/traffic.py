import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])

    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


# =========================
# LOAD DATA
# =========================
def load_data(data_dir):
    images = []
    labels = []

    for label in range(NUM_CATEGORIES):
        folder = os.path.join(data_dir, str(label))

        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            image = cv2.imread(file_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


# =========================
# BUILD MODEL
# =========================
def get_model():

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            32, (3, 3),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()