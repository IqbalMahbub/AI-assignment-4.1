# Assignment 16
#----------------------------------------------------------------------------------------
# Write a report describing how monitoring performance curves for both the training set
# and the validation set based on the target metric (e.g., ‘accuracy’) and ‘loss’ metric can
# improve your hyperparameter training.
#----------------------------------------------------------------------------------------


#Monitoring Performance Curves
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model


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

def build_model(input_shape, filters=32, dense_units=128):

    inputs = Input(shape=input_shape)

    x = Conv2D(filters, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(filters*2, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train, epochs=15):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Curves
# --------------------------------------------------

def plot_curves(history, name):

    # Loss Curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(f"{name}_loss.png")
    plt.show()

    # Accuracy Curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(f"{name}_accuracy.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    # Example Hyperparameters
    model = build_model(
        input_shape=x_train.shape[1:],
        filters=32,
        dense_units=128
    )

    history = train_model(model, x_train, y_train, epochs=15)

    model.evaluate(x_test, y_test)

    plot_curves(history, "performance_monitoring")


if __name__ == "__main__":
    main()

