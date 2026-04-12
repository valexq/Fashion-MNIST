"""
predict.py
----------
Script CLI para clasificar una imagen real de prenda usando un modelo entrenado.

Uso:
    python predict.py --image <ruta_imagen> --model <mlp|cnn> [--invert]

Ejemplos:
    python predict.py --image foto_camiseta.jpg --model cnn
    python predict.py --image foto_zapatilla.jpg --model cnn --invert

Nota: por defecto se usa auto-detección del fondo (invert=None).
      Usa --invert solo si la auto-detección falla.
"""

import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from datos_processing import (
    class_names,
    prepare_real_image_for_mlp,
    prepare_real_image_for_cnn
)

# Ruta al directorio de modelos (relativa a este script)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

# Nombres de modelo (formato .keras moderno; fallback a .h5 si no existe)
def _model_path(name: str) -> str:
    p = os.path.join(MODELS_DIR, f'{name}_model.keras')
    if not os.path.exists(p):
        p = os.path.join(MODELS_DIR, f'{name}_model.h5')  # compatibilidad legacy
    return p


def predict_image(image_path: str, model_type: str, invert: bool = False):
    """
    Carga el modelo entrenado y predice la clase de una imagen real.

    Parámetros
    ----------
    image_path : str
        Ruta al archivo de imagen (JPG, PNG, etc.)
    model_type : str
        Tipo de modelo a usar: 'mlp' o 'cnn'
    invert : bool
        Si True, invierte los colores de la imagen antes de clasificar
        (útil cuando el fondo de la foto es oscuro).

    Retorna
    -------
    str
        Nombre de la clase predicha.
    """
    model_type = model_type.lower()

    if model_type == 'mlp':
        model_path = _model_path('mlp')
        img_array = prepare_real_image_for_mlp(image_path, invert=invert)
    elif model_type == 'cnn':
        model_path = _model_path('cnn')
        img_array = prepare_real_image_for_cnn(image_path, invert=invert)
    else:
        raise ValueError(f"Tipo de modelo no reconocido: '{model_type}'. Use 'mlp' o 'cnn'.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado en: {model_path}\n"
            f"Ejecuta primero: python src/train_{model_type}.py"
        )

    print(f"Cargando modelo '{model_type}' desde: {model_path}")
    model = load_model(model_path)

    # Predicción
    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index] * 100

    # Mostrar resultados
    print("\n" + "=" * 45)
    print(f"  Predicción: {predicted_class}")
    print(f"  Confianza:  {confidence:.1f}%")
    print("=" * 45)

    # Top-3 probabilidades
    top3_indices = np.argsort(probabilities)[::-1][:3]
    print("\nTop-3 clases:")
    for rank, idx in enumerate(top3_indices, start=1):
        bar_len = int(probabilities[idx] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {rank}. {class_names[idx]:<15} [{bar}] {probabilities[idx]*100:5.1f}%")
    print()

    return predicted_class


def main():
    parser = argparse.ArgumentParser(
        description="Clasifica una imagen de prenda de vestir con Fashion MNIST."
    )
    parser.add_argument(
        '--image', '-i',
        required=True,
        help='Ruta a la imagen a clasificar (JPG, PNG, etc.)'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        choices=['mlp', 'cnn'],
        help='Modelo a usar: mlp o cnn'
    )
    parser.add_argument(
        '--invert',
        action='store_true',
        help=(
            'Forzar inversión de colores. Por defecto se usa auto-detección.\n'
            'Usa esta opción solo si la clasificación parece incorrecta.'
        )
    )
    args = parser.parse_args()

    # None = auto-detectar (recomendado); True = forzar inversión
    invert_mode = True if args.invert else None

    predict_image(
        image_path=args.image,
        model_type=args.model,
        invert=invert_mode
    )


if __name__ == "__main__":
    main()
