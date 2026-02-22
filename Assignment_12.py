# Assignment 12
#----------------------------------------------------------------------------------------------
# Write a report by discussing the effect of different data augmentation techniques on your
# CNN based classifiers.
#----------------------------------------------------------------------------------------------
#Data Augmentation Effect (Code)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build CNN Model
# --------------------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------------------------
# Train Without Augmentation
# --------------------------------------------------

def train_without_aug(model, x_train, y_train):

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Train With Augmentation
# --------------------------------------------------

def train_with_aug(model, x_train, y_train):

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    datagen.fit(x_train)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=5,
        validation_data=(x_train[:6000], y_train[:6000]),
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Comparison
# --------------------------------------------------

def plot_comparison(history1, history2):

    # Loss
    plt.figure()
    plt.plot(history1.history['val_loss'], label='No Aug Val Loss')
    plt.plot(history2.history['val_loss'], label='Aug Val Loss')
    plt.legend()
    plt.title("Validation Loss Comparison")
    plt.savefig("augmentation_loss.png")
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(history1.history['val_accuracy'], label='No Aug Val Acc')
    plt.plot(history2.history['val_accuracy'], label='Aug Val Acc')
    plt.legend()
    plt.title("Validation Accuracy Comparison")
    plt.savefig("augmentation_accuracy.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    # -------------------------
    # Without Augmentation
    # -------------------------
    print("\nTraining WITHOUT augmentation")
    model1 = build_model(x_train.shape[1:])
    history1 = train_without_aug(model1, x_train, y_train)
    model1.evaluate(x_test, y_test)

    # -------------------------
    # With Augmentation
    # -------------------------
    print("\nTraining WITH augmentation")
    model2 = build_model(x_train.shape[1:])
    history2 = train_with_aug(model2, x_train, y_train)
    model2.evaluate(x_test, y_test)

    # -------------------------
    # Plot Comparison
    # -------------------------
    plot_comparison(history1, history2)


if __name__ == "__main__":
    main()
