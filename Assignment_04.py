# Assignment 4
#---------------------------------------------------------------------------------------------
# Building an FCFNN based classifier according to your preferences about
# the number of hidden layers and neurons in the hidden layers.
# • training and testing your FCFNN based classifier using the:
# i. Fashion MNIST dataset.
# ii. MNIST English dataset.
# iii. CIFAR-10 dataset.
#-----------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(dataset_name):

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build FCFNN Model
# --------------------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape, name="input_layer")

    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name="FCFNN_Classifier")

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Loss
# --------------------------------------------------

def plot_loss(history, name):

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss Curve")
    plt.legend()

    plt.savefig(f"{name}_loss.png")
    plt.show()


# --------------------------------------------------
# Plot 10 Predictions
# --------------------------------------------------

def plot_predictions(model, x_test, y_test, name):

    predictions = model.predict(x_test[:10])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10,4))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i], cmap='gray' if len(x_test.shape)==3 else None)
        plt.title(f"P:{predicted_labels[i]}\nT:{y_test[i]}")
        plt.axis('off')

    plt.suptitle(f"{name} Predictions")
    plt.savefig(f"{name}_predictions.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for dataset in ["mnist", "fashion", "cifar10"]:

        print("\n====================================")
        print(f"Training on {dataset.upper()}")
        print("====================================")

        x_train, x_test, y_train, y_test = load_data(dataset)

        model = build_model(input_shape=x_train.shape[1:])

        model.summary()

        history = train_model(model, x_train, y_train)

        model.evaluate(x_test, y_test, verbose=1)

        plot_loss(history, dataset)

        plot_predictions(model, x_test, y_test, dataset)


if __name__ == "__main__":
    main()
