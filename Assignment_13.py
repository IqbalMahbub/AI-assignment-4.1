# Assignment 13
#-------------------------------------------------------------------------------------------
# Show the effect of dropout layer, data augmentation techniques on overfitting issues of
# your CNN based classifier.
#-------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout
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

def build_model(input_shape, use_dropout=False):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------------------------
# Train Without Augmentation
# --------------------------------------------------

def train_normal(model, x_train, y_train):

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

def plot_results(h1, h2, h3):

    # Validation Loss
    plt.figure()
    plt.plot(h1.history['val_loss'], label='Normal')
    plt.plot(h2.history['val_loss'], label='Dropout')
    plt.plot(h3.history['val_loss'], label='Dropout + Aug')
    plt.legend()
    plt.title("Validation Loss Comparison")
    plt.savefig("overfitting_loss.png")
    plt.show()

    # Validation Accuracy
    plt.figure()
    plt.plot(h1.history['val_accuracy'], label='Normal')
    plt.plot(h2.history['val_accuracy'], label='Dropout')
    plt.plot(h3.history['val_accuracy'], label='Dropout + Aug')
    plt.legend()
    plt.title("Validation Accuracy Comparison")
    plt.savefig("overfitting_accuracy.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    # -----------------------
    # 1. Normal CNN
    # -----------------------
    print("\nTraining Normal CNN")
    model1 = build_model(x_train.shape[1:], use_dropout=False)
    h1 = train_normal(model1, x_train, y_train)

    # -----------------------
    # 2. CNN with Dropout
    # -----------------------
    print("\nTraining CNN with Dropout")
    model2 = build_model(x_train.shape[1:], use_dropout=True)
    h2 = train_normal(model2, x_train, y_train)

    # -----------------------
    # 3. CNN with Dropout + Augmentation
    # -----------------------
    print("\nTraining CNN with Dropout + Augmentation")
    model3 = build_model(x_train.shape[1:], use_dropout=True)
    h3 = train_with_aug(model3, x_train, y_train)

    plot_results(h1, h2, h3)


if __name__ == "__main__":
    main()
