"""
app.py
------
Interfaz web interactiva para el sistema de reconocimiento de prendas Fashion MNIST.
Permite subir una imagen real y clasificarla con el modelo MLP o CNN entrenado.

Uso:
    python app.py

Luego abre el navegador en: http://localhost:7860
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model
from datos_processing import (
    class_names,
    preprocess_real_image,
    prepare_real_image_for_mlp,
    prepare_real_image_for_cnn
)

# ---------------------------------------------------------------------------
# Rutas — soporta tanto .keras (nuevo) como .h5 (legacy)
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
TMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tmp_upload.png')


def _model_path(name: str) -> str:
    """Busca el modelo en formato .keras primero, luego .h5 como fallback."""
    for ext in ('.keras', '.h5'):
        p = os.path.join(MODELS_DIR, f'{name}_model{ext}')
        if os.path.exists(p):
            return p
    return os.path.join(MODELS_DIR, f'{name}_model.keras')


# ---------------------------------------------------------------------------
# Caché de modelos (se cargan una sola vez al primer uso)
# ---------------------------------------------------------------------------
_models: dict = {}


def _get_model(model_type: str):
    """Carga y cachea el modelo."""
    if model_type not in _models:
        path = _model_path(model_type.lower())
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modelo '{model_type}' no encontrado.\n"
                f"Ejecuta: python src/train_{model_type.lower()}.py"
            )
        _models[model_type] = load_model(path)
    return _models[model_type]


# ---------------------------------------------------------------------------
# Función de predicción
# ---------------------------------------------------------------------------
def classify(image, model_choice: str, force_invert: bool):
    """
    Clasifica la imagen y retorna:
    - Probabilidades por clase (dict)
    - Vista previa 28×28 de lo que el modelo recibe (numpy array)
    """
    empty_probs = {c: 0.0 for c in class_names}
    empty_preview = np.zeros((28, 28), dtype='uint8')

    if image is None:
        return empty_probs, empty_preview

    # Guardar temporalmente la imagen subida
    image.save(TMP_PATH)

    # invert=None → auto-detectar; True → forzar inversión
    invert_mode = True if force_invert else None

    try:
        model = _get_model(model_choice)

        # Obtener la imagen preprocesada de 28×28 (lo que el modelo "ve")
        processed_28x28 = preprocess_real_image(TMP_PATH, invert=invert_mode)

        # Preparar para el modelo según el tipo
        if model_choice == 'MLP':
            img_input = processed_28x28.reshape(1, 28 * 28)
        else:
            img_input = processed_28x28.reshape(1, 28, 28, 1)

        probs = model.predict(img_input, verbose=0)[0]
        result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        # Escalar la vista previa a 112×112 para que sea visible
        preview_uint8 = (processed_28x28 * 255).astype('uint8')
        preview_img = Image.fromarray(preview_uint8, mode='L')
        preview_img = preview_img.resize((112, 112), Image.NEAREST)
        preview_array = np.array(preview_img)

    except FileNotFoundError as e:
        return {"⚠️ " + str(e): 1.0}, empty_preview
    except Exception as e:
        return {"⚠️ Error: " + str(e): 1.0}, empty_preview
    finally:
        if os.path.exists(TMP_PATH):
            os.remove(TMP_PATH)

    return result, preview_array


# ---------------------------------------------------------------------------
# Layout de la interfaz Gradio
# ---------------------------------------------------------------------------
DESCRIPTION = """
## 👗 Reconocimiento de Prendas de Vestir — Fashion MNIST

Sube una foto de una prenda y el sistema la clasificará en una de las **10 categorías** del dataset.

> ℹ️ La imagen se transforma automáticamente al formato Fashion MNIST (fondo negro, prenda clara, 28×28 px).
> Revisa la **"Vista previa"** para ver exactamente lo que el modelo recibe.
"""

with gr.Blocks(title="Fashion MNIST — Reconocimiento de Prendas", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # --- Columna izquierda: controles ---
        with gr.Column(scale=1):
            image_input = gr.Image(
                type='pil',
                label='📷 Sube tu imagen aquí',
                height=300
            )

            with gr.Row():
                model_choice = gr.Radio(
                    choices=['MLP', 'CNN'],
                    value='CNN',
                    label='🤖 Modelo',
                    info='CNN = mayor precisión | MLP = red básica'
                )
                force_invert = gr.Checkbox(
                    label='🔄 Forzar inversión',
                    value=False,
                    info='Activa si la vista previa no se ve correcta'
                )

            btn = gr.Button('🔍 Clasificar prenda', variant='primary', size='lg')

        # --- Columna central: vista previa ---
        with gr.Column(scale=1):
            preview_output = gr.Image(
                label='🔬 Vista previa: lo que el modelo "ve" (28×28 px)',
                height=180,
                type='numpy'
            )
            gr.Markdown("""
> **¿La vista previa se ve bien?**
> - La prenda debe verse **clara/blanca** sobre **fondo negro**.
> - Si la prenda aparece como una mancha negra, activa **"Forzar inversión"**.
> - Si se ve muy borroso, la foto puede estar muy lejos o tener poco contraste.
""")

        # --- Columna derecha: resultados ---
        with gr.Column(scale=1):
            label_output = gr.Label(
                num_top_classes=10,
                label='📊 Probabilidades por clase'
            )

    # Eventos
    btn.click(
        fn=classify,
        inputs=[image_input, model_choice, force_invert],
        outputs=[label_output, preview_output]
    )

    image_input.change(
        fn=classify,
        inputs=[image_input, model_choice, force_invert],
        outputs=[label_output, preview_output]
    )

    gr.Markdown("""
---
### 📸 Consejos para mejores resultados
- 📷 Usa **fondo blanco o claro** uniforme (una hoja, una mesa clara).
- 👕 Extiende la prenda **plana** y centrada.
- 💡 Buena **iluminación** sin sombras fuertes.
- 🔄 Si el resultado es incorrecto, activa **"Forzar inversión"** y mira la vista previa.

### 🏷️ Las 10 clases del dataset

| # | Clase | # | Clase |
|---|-------|---|-------|
| 0 | 👕 T-shirt/top | 5 | 👡 Sandal |
| 1 | 👖 Trouser | 6 | 👔 Shirt |
| 2 | 🧥 Pullover | 7 | 👟 Sneaker |
| 3 | 👗 Dress | 8 | 👜 Bag |
| 4 | 🧥 Coat | 9 | 👢 Ankle boot |
""")

if __name__ == '__main__':
    demo.launch(share=False)
