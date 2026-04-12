import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from datos_processing import load_and_prepare_all

# Ruta absoluta a la carpeta models/ (relativa a este script)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def build_mlp():
    """
    Red neuronal densa (MLP) mejorada para clasificación Fashion MNIST.
    Arquitectura: 784 → 256 → 128 → 64 → 10
    """
    model = Sequential([
        # Capa de entrada + primera capa oculta
        Dense(256, activation='relu', input_shape=(784,)),
        BatchNormalization(),
        Dropout(0.3),

        # Segunda capa oculta
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Tercera capa oculta
        Dense(64, activation='relu'),
        Dropout(0.2),

        # Capa de salida
        Dense(10, activation='softmax')
    ])
    return model


def main():
    data = load_and_prepare_all()

    x_train = data['x_train_mlp']
    y_train = data['y_train']
    x_test  = data['x_test_mlp']
    y_test  = data['y_test']

    print("Shape x_train:", x_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape x_test :", x_test.shape)
    print("Shape y_test :", y_test.shape)

    model = build_mlp()
    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nLoss en test:     {test_loss:.4f}")
    print(f"Accuracy en test: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    save_path = os.path.join(MODELS_DIR, 'mlp_model.keras')
    model.save(save_path)
    print(f"Modelo MLP guardado en {save_path}")


if __name__ == '__main__':
    main()