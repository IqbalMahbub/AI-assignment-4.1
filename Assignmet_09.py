import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os

# ---------------------------
# SETTINGS
# ---------------------------
IMG_PATH = "download.jpg"
OUTPUT_DIR = "featuremaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def load_and_preprocess(img_path, target_size=(224,224), model_type="vgg"):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if model_type == "vgg":
        x = vgg_preprocess(x)
    elif model_type == "resnet":
        x = resnet_preprocess(x)
    elif model_type == "inception":
        x = inception_preprocess(x)
    return x

def save_feature_maps(feature_maps, output_prefix, max_maps=16):
    """Save the first max_maps feature maps as images"""
    feature_maps = feature_maps[0]  # remove batch dimension
    n_maps = min(feature_maps.shape[-1], max_maps)
    plt.figure(figsize=(15, 8))

    for i in range(n_maps):
        plt.subplot(4, 4, i+1)
        plt.imshow(feature_maps[:, :, i], cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{output_prefix}.png")
    plt.show()
    plt.close()

# ---------------------------
# LOAD MODELS
# ---------------------------
vgg = VGG16(weights='imagenet', include_top=False)
resnet = ResNet50(weights='imagenet', include_top=False)
inception = InceptionV3(weights='imagenet', include_top=False)

# Select layers to visualize (you can change indices)
vgg_layers = [0, 3, 5]         # conv layers for VGG16
resnet_layers = [2, 5, 10]     # conv layers for ResNet50
inception_layers = [1, 5, 10]  # conv layers for InceptionV3

# ---------------------------
# VGG16 FEATURE MAPS
# ---------------------------
x_vgg = load_and_preprocess(IMG_PATH, target_size=(224,224), model_type="vgg")
for idx in vgg_layers:
    layer_model = Model(inputs=vgg.input, outputs=vgg.layers[idx].output)
    feature_maps = layer_model.predict(x_vgg)
    save_feature_maps(feature_maps, f"vgg16_layer{idx}")

# ---------------------------
# ResNet50 FEATURE MAPS
# ---------------------------
x_resnet = load_and_preprocess(IMG_PATH, target_size=(224,224), model_type="resnet")
for idx in resnet_layers:
    layer_model = Model(inputs=resnet.input, outputs=resnet.layers[idx].output)
    feature_maps = layer_model.predict(x_resnet)
    save_feature_maps(feature_maps, f"resnet50_layer{idx}")

# ---------------------------
# InceptionV3 FEATURE MAPS
# ---------------------------
x_inception = load_and_preprocess(IMG_PATH, target_size=(299,299), model_type="inception")
for idx in inception_layers:
    layer_model = Model(inputs=inception.input, outputs=inception.layers[idx].output)
    feature_maps = layer_model.predict(x_inception)
    save_feature_maps(feature_maps, f"inceptionv3_layer{idx}")
