# Assignment 10
#---------------------------------------------------------------------------
# Write a report in pdf format using any Latex system after:
# ● training a binary classifier, based on the pre-trained VGG16, by transfer learning
# and fine tuning.
# ● showing the effect of fine-tuning:
# i. whole pre-trained VGG16
# ii. partial pre-trained VGG16
#--------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# --------------------------------------------------
# Load Dataset (Binary: Cats vs Dogs)
# --------------------------------------------------

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keep only class 3 (cat) and 5 (dog)
    train_filter = np.where((y_train==3) | (y_train==5))[0]
    test_filter = np.where((y_test==3) | (y_test==5))[0]

    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    y_train = (y_train==5).astype(int)
    y_test = (y_test==5).astype(int)

    x_train = tf.image.resize(x_train, (224,224))
    x_test = tf.image.resize(x_test, (224,224))

    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build Model
# --------------------------------------------------

def build_model(trainable_layers=None):

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

    if trainable_layers is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    inputs = Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Results
# --------------------------------------------------

def plot_history(history, name):

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss")
    plt.legend()
    plt.savefig(f"{name}_loss.png")
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.savefig(f"{name}_accuracy.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    # -------------------------------
    # Case 1: Freeze Whole VGG16
    # -------------------------------
    print("\n=== Transfer Learning (Freeze All Layers) ===")
    model1 = build_model(trainable_layers=None)
    history1 = train_model(model1, x_train, y_train)
    model1.evaluate(x_test, y_test)
    plot_history(history1, "freeze_all")


    # -------------------------------
    # Case 2: Partial Fine-Tuning
    # -------------------------------
    print("\n=== Partial Fine-Tuning (Last 4 Layers Trainable) ===")
    model2 = build_model(trainable_layers=4)
    history2 = train_model(model2, x_train, y_train)
    model2.evaluate(x_test, y_test)
    plot_history(history2, "partial_finetune")


    # -------------------------------
    # Case 3: Fine-Tune Whole VGG16
    # -------------------------------
    print("\n=== Fine-Tune Whole VGG16 ===")
    model3 = build_model(trainable_layers=len(VGG16().layers))
    history3 = train_model(model3, x_train, y_train)
    model3.evaluate(x_test, y_test)
    plot_history(history3, "full_finetune")


if __name__ == "__main__":
    main()
