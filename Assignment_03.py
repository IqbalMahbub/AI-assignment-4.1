import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

def generate_data(power):

    x = np.linspace(-10, 10, 2000)

    if power == 1:
        y = 5*x + 10

    elif power == 2:
        y = 3*(x**2) + 5*x + 10

    elif power == 3:
        y = 4*(x**3) + 3*(x**2) + 5*x + 10

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return train_test_split(x, y, test_size=0.3, random_state=42)


# --------------------------------------------------
# Build Model
# --------------------------------------------------

def build_model():

    inputs = Input(shape=(1,), name="input_layer")

    h1 = Dense(32, activation='relu')(inputs)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)

    outputs = Dense(1, activation='linear')(h3)

    model = Model(inputs, outputs, name="FCFNN")

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )


# --------------------------------------------------
# Plot Results
# --------------------------------------------------

def plot_results(model, x, y, title, filename):

    y_pred = model.predict(x)

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, label="Original y")
    plt.scatter(x, y_pred, label="Predicted y")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.savefig(filename)
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for power in [1, 2, 3]:

        print("\n===================================")
        print(f"Training for Power {power}")
        print("===================================")

        x_train, x_test, y_train, y_test = generate_data(power)

        model = build_model()

        model.summary()

        train_model(model, x_train, y_train)

        plot_results(
            model,
            x_test,
            y_test,
            title=f"Polynomial Power {power}",
            filename=f"power_{power}.png"
        )


if __name__ == "__main__":
    main()