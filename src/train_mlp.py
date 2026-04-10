from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datos_processing import load_and_prepare_all  # o desde el nombre que tengas

def main():
    data = load_and_prepare_all()

    x_train = data["x_train_mlp"]
    y_train = data["y_train"]
    x_test = data["x_test_mlp"]
    y_test = data["y_test"]

    print("Shape x_train:", x_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape x_test :", x_test.shape)
    print("Shape y_test :", y_test.shape)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(784,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss en test: {test_loss:.4f}")
    print(f"Accuracy en test: {test_acc:.4f}")

    model.save("models/mlp_model.h5")
    print("Modelo MLP guardado en models/mlp_model.h5")

if __name__ == "__main__":
    main()