# Assignment 11
#--------------------------------------------------------------------------------------------
# Discuss the feature extraction power of your favorite CNN pretrained by the ImageNet
# dataset before and after transfer learning by the MNIST digit dataset after plotting high
# dimensional feature vectors on 2D plane using the following two dimension reduction
# techniques:
# ● Principal Component Analysis (PCA)
# ● t-distributed Stochastic Neighbor Embedding (t-SNE)
#--------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --------------------------------------------------
# Load MNIST
# --------------------------------------------------

def load_data():

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:2000]          # small subset for speed
    y_train = y_train[:2000]

    x_train = np.expand_dims(x_train, -1)
    x_train = np.repeat(x_train, 3, axis=-1)

    x_train = tf.image.resize(x_train, (224,224))
    x_train = preprocess_input(x_train)

    return x_train, y_train


# --------------------------------------------------
# Build Feature Extractor (Before Transfer Learning)
# --------------------------------------------------

def build_feature_extractor(trainable=False):

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224,224,3))

    base_model.trainable = trainable

    inputs = Input(shape=(224,224,3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs, x)

    return model


# --------------------------------------------------
# Fine-Tune Model on MNIST
# --------------------------------------------------

def fine_tune_model(x_train, y_train):

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(224,224,3))

    base_model.trainable = True

    inputs = Input(shape=(224,224,3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=3,
              batch_size=32,
              verbose=1)

    return model


# --------------------------------------------------
# Extract Features
# --------------------------------------------------

def extract_features(model, x):

    features = model.predict(x, batch_size=32)
    return features


# --------------------------------------------------
# PCA Plot
# --------------------------------------------------

def plot_pca(features, labels, name):

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure()
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels)
    plt.title(f"{name} PCA")
    plt.colorbar(scatter)

    plt.savefig(f"{name}_pca.png")
    plt.show()


# --------------------------------------------------
# t-SNE Plot
# --------------------------------------------------

def plot_tsne(features, labels, name):

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure()
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels)
    plt.title(f"{name} t-SNE")
    plt.colorbar(scatter)

    plt.savefig(f"{name}_tsne.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, y_train = load_data()

    # -----------------------------
    # Before Transfer Learning
    # -----------------------------
    print("Extracting features BEFORE fine-tuning...")
    feature_model_before = build_feature_extractor(trainable=False)
    features_before = extract_features(feature_model_before, x_train)

    plot_pca(features_before, y_train, "before_transfer")
    plot_tsne(features_before, y_train, "before_transfer")


    # -----------------------------
    # After Transfer Learning
    # -----------------------------
    print("Fine-tuning on MNIST...")
    fine_tuned_model = fine_tune_model(x_train, y_train)

    feature_model_after = Model(
        fine_tuned_model.input,
        fine_tuned_model.layers[-3].output
    )

    features_after = extract_features(feature_model_after, x_train)

    plot_pca(features_after, y_train, "after_transfer")
    plot_tsne(features_after, y_train, "after_transfer")


if __name__ == "__main__":
    main()
