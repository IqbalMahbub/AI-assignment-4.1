# Assignment 5
#-----------------------------------------------------------------------------
# Building a Convolutional Neural Network (CNN) based 10 class classifier
# • training and testing the classifier using the:
# i. Fashion MNIST dataset.
# ii. MNIST English dataset.
# iii. CIFAR-10 dataset.
#----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(dataset_name):

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build CNN Model
# --------------------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape, name="input_layer")

    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name="CNN_Classifier")

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

    plt.savefig(f"{name}_cnn_loss.png")
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
        plt.imshow(x_test[i].reshape(28,28), cmap='gray')
        plt.title(f"P:{predicted_labels[i]}\nT:{y_test[i]}")
        plt.axis('off')

    plt.suptitle(f"{name} CNN Predictions")
    plt.savefig(f"{name}_cnn_predictions.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for dataset in ["mnist", "fashion"]:

        print("\n====================================")
        print(f"Training CNN on {dataset.upper()}")
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
