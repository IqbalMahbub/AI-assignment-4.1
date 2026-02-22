# Assignment 14
#----------------------------------------------------------------------------------------
# Write a report by discussing the effect of the following issues on the classifier’s
# performance:
# ● different activation functions in hidden layers
# ● different loss functions
#----------------------------------------------------------------------------------------
#Activation & Loss Function Comparison

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test, y_train_cat, y_test_cat


# --------------------------------------------------
# Build CNN
# --------------------------------------------------

def build_model(input_shape, activation_function):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation=activation_function)(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation=activation_function)(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(128, activation=activation_function)(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train, loss_function):

    model.compile(optimizer='adam',
                  loss=loss_function,
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
# Plot Comparison
# --------------------------------------------------

def plot_activation_results(histories):

    plt.figure()
    for name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=name)

    plt.title("Activation Function Comparison")
    plt.legend()
    plt.savefig("activation_comparison.png")
    plt.show()


def plot_loss_results(histories):

    plt.figure()
    for name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=name)

    plt.title("Loss Function Comparison")
    plt.legend()
    plt.savefig("loss_comparison.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test, y_train_cat, y_test_cat = load_data()

    # -----------------------------
    # Activation Function Comparison
    # -----------------------------
    activation_histories = {}

    for activation in ["relu", "tanh", "sigmoid"]:

        print(f"\nTraining with {activation} activation")
        model = build_model(x_train.shape[1:], activation)
        history = train_model(model, x_train, y_train, "sparse_categorical_crossentropy")
        activation_histories[activation] = history

    plot_activation_results(activation_histories)


    # -----------------------------
    # Loss Function Comparison
    # -----------------------------
    loss_histories = {}

    print("\nTraining with Sparse Categorical Crossentropy")
    model1 = build_model(x_train.shape[1:], "relu")
    history1 = train_model(model1, x_train, y_train, "sparse_categorical_crossentropy")
    loss_histories["Sparse_CE"] = history1

    print("\nTraining with Categorical Crossentropy")
    model2 = build_model(x_train.shape[1:], "relu")
    history2 = train_model(model2, x_train, y_train_cat, "categorical_crossentropy")
    loss_histories["Categorical_CE"] = history2

    plot_loss_results(loss_histories)


if __name__ == "__main__":
    main()
