import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datos_processing import load_and_prepare_all

# Ruta absoluta a la carpeta models/ (relativa a este script)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def build_cnn():
    """
    Red Neuronal Convolucional (CNN) para Fashion MNIST.

    Arquitectura:
      Bloque 1: Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
      Bloque 2: Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
      Clasificador: Flatten → Dense(256) → BN → Dropout(0.5) → Dense(10)
    """
    model = Sequential([
        # Bloque convolucional 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Bloque convolucional 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Clasificador
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model


def main():
    data = load_and_prepare_all()

    x_train_full = data['x_train_cnn']
    y_train_full = data['y_train']
    x_test       = data['x_test_cnn']
    y_test       = data['y_test']

    # --- Separar validación antes del augmentation ---
    val_size = int(len(x_train_full) * 0.1)
    x_val,   y_val   = x_train_full[:val_size], y_train_full[:val_size]
    x_train, y_train = x_train_full[val_size:],  y_train_full[val_size:]

    print("Shape x_train:", x_train.shape)
    print("Shape x_val  :", x_val.shape)
    print("Shape x_test :", x_test.shape)

    # --- Data Augmentation ---
    # Aplica transformaciones aleatorias durante el entrenamiento para que el
    # modelo sea más robusto ante variaciones de fotos reales (ángulo, zoom, posición).
    # NO se aplica a validación ni a test.
    datagen = ImageDataGenerator(
        rotation_range=10,        # rotación aleatoria ±10°
        width_shift_range=0.1,    # desplazamiento horizontal ±10%
        height_shift_range=0.1,   # desplazamiento vertical ±10%
        zoom_range=0.1,           # zoom ±10%
        fill_mode='nearest'       # rellena pixels nuevos con el valor vecino más cercano
    )
    datagen.fit(x_train)

    model = build_cnn()
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

    steps_per_epoch = len(x_train) // 64

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=steps_per_epoch,
        epochs=25,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nLoss en test:     {test_loss:.4f}")
    print(f"Accuracy en test: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    save_path = os.path.join(MODELS_DIR, 'cnn_model.keras')
    model.save(save_path)
    print(f"Modelo CNN guardado en {save_path}")


if __name__ == '__main__':
    main()
