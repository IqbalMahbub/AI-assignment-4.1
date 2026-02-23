import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets.mnist import load_data

def build_model():
    """Defines a Functional API model for MNIST classification."""
    inputs = Input(shape=(28, 28))
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

def main():
    # --- 1. Load Custom Dataset ---
    # Ensure 'mnist1.npz' exists in your working directory
    custom = np.load('mnist1.npz')
    custom_trainX = custom['trainX'].astype('float32') / 255.0
    custom_trainY = to_categorical(custom['trainY'], 10)
    custom_testX = custom['testX'].astype('float32') / 255.0
    custom_testY = to_categorical(custom['testY'], 10)

    # --- 2. Load Standard MNIST Dataset ---
    (mnist_trainX, mnist_trainY), (mnist_testX, mnist_testY) = load_data()
    mnist_trainX = mnist_trainX.astype('float32') / 255.0
    mnist_testX = mnist_testX.astype('float32') / 255.0
    mnist_trainY = to_categorical(mnist_trainY, 10)
    mnist_testY = to_categorical(mnist_testY, 10)

    # --- 3. Combine Datasets for Evaluation ---
    combined_testX = np.concatenate([mnist_testX, custom_testX], axis=0)
    combined_testY = np.concatenate([mnist_testY, custom_testY], axis=0)

    print(f"Custom Train: {custom_trainX.shape}")
    print(f"MNIST Train:  {mnist_trainX.shape}")
    print(f"Combined Test: {combined_testX.shape}")

    # --- 4. Phase 1: Train on Custom Data ---
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\n--- Starting Phase 1: Custom Data Training ---")
    model.fit(custom_trainX, custom_trainY, validation_split=0.1, epochs=10, batch_size=32)

    # --- 5. Phase 2: Fine-tune on MNIST ---
    print("\n--- Starting Phase 2: MNIST Fine-tuning ---")
    history = model.fit(mnist_trainX, mnist_trainY, validation_split=0.1, epochs=10, batch_size=64)

    # --- 6. Final Evaluation ---
    mnist_acc = model.evaluate(mnist_testX, mnist_testY, verbose=0)[1]
    custom_acc = model.evaluate(custom_testX, custom_testY, verbose=0)[1]
    combined_acc = model.evaluate(combined_testX, combined_testY, verbose=0)[1]

    print(f"\nFinal MNIST Test Acc:  {mnist_acc:.4f}")
    print(f"Final Custom Test Acc: {custom_acc:.4f}")
    print(f"Combined Test Acc:     {combined_acc:.4f}")

    # --- 7. Plotting Results ---
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('MNIST Phase Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('MNIST Phase Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()