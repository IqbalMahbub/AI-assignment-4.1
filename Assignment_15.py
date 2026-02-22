# Assignment 15
#--------------------------------------------------------------------------------------------
# Write a report by describing how different callback functions can make your training
# process better.
#--------------------------------------------------------------------------------------------

#Effect of Callback Functions on Training Process.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


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
# Train Model with Callbacks
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor='val_accuracy',
        save_best_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Results
# --------------------------------------------------

def plot_results(history):

    # Loss Curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss with Callbacks")
    plt.savefig("callbacks_loss.png")
    plt.show()

    # Accuracy Curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy with Callbacks")
    plt.savefig("callbacks_accuracy.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    model = build_model(x_train.shape[1:])

    history = train_model(model, x_train, y_train)

    model.evaluate(x_test, y_test)

    plot_results(history)


if __name__ == "__main__":
    main()
