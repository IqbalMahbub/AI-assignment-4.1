# Fully Connected Feed-Forward Neural Network (FCFNN)
# Architecture: Input(8) → 4 → 8 → 4 → 10(Output)


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def build_model():

    inputs = Input(shape=(8,), name="input_layer")

    h1 = Dense(4, activation='relu', name='hidden_layer1')(inputs)
    h2 = Dense(8, activation='relu', name='hidden_layer2')(h1)
    h3 = Dense(4, activation='relu', name='hidden_layer3')(h2)
    outputs = Dense(10, activation='softmax', name='output_layer')(h3)
    model = Model(inputs, outputs, name="FCFNN_Functional")


    return model

def main():

    model = build_model()
    model.summary(show_trainable=True)

if __name__ == "__main__":
    main()
