import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tensorflow.keras.datasets import fashion_mnist

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# ---------------------------------------------------------------------------
# Dataset Fashion MNIST
# ---------------------------------------------------------------------------

def load_fashion_mnist():
    """Carga el dataset Fashion MNIST desde TensorFlow."""
    return fashion_mnist.load_data()


def normalize_images(x_train, x_test):
    """Normaliza los píxeles al rango [0, 1]."""
    return x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0


def prepare_for_mlp(images):
    """Convierte imágenes (N, 28, 28) → (N, 784) para la red densa."""
    return images.reshape(images.shape[0], 28 * 28)


def prepare_for_cnn(images):
    """Convierte imágenes (N, 28, 28) → (N, 28, 28, 1) para la CNN."""
    return images.reshape(images.shape[0], 28, 28, 1)


def load_and_prepare_all():
    """
    Carga el dataset completo, lo normaliza y devuelve un diccionario con
    las versiones listas para MLP y CNN.
    """
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    x_train, x_test = normalize_images(x_train, x_test)

    return {
        'x_train':     x_train,
        'y_train':     y_train,
        'x_test':      x_test,
        'y_test':      y_test,
        'x_train_mlp': prepare_for_mlp(x_train),
        'x_test_mlp':  prepare_for_mlp(x_test),
        'x_train_cnn': prepare_for_cnn(x_train),
        'x_test_cnn':  prepare_for_cnn(x_test),
        'class_names': class_names
    }


# ---------------------------------------------------------------------------
# Preprocesamiento de imágenes REALES
# ---------------------------------------------------------------------------
#
# Las imágenes de Fashion MNIST tienen estas características:
#   • 28×28 px, escala de grises
#   • Fondo NEGRO (valor ~0)
#   • Prenda en tonos de GRIS/BLANCO
#   • Prenda centrada, sin rotación extrema
#   • Aspecto de "silueta" con detalles de textura suaves
#
# Las fotos reales son muy diferentes:
#   • Alta resolución, color
#   • Fondo claro (blanco/gris/color)
#   • La prenda puede estar en cualquier posición
#   • Hay sombras, pliegues, texturas reales
#
# Para cerrar este "domain gap", este pipeline transforma la foto real
# para que se parezca LO MÁS POSIBLE a una imagen de Fashion MNIST.
# ---------------------------------------------------------------------------

def preprocess_real_image(image_path, invert=None):
    """
    Convierte una imagen real al formato Fashion MNIST con preprocesamiento
    agresivo para maximizar la precisión del modelo.

    Parámetros
    ----------
    image_path : str
        Ruta a la imagen (JPG, PNG, etc.).
    invert : bool | None
        - None  → auto-detectar según el brillo del fondo (RECOMENDADO).
        - True  → forzar inversión.
        - False → nunca invertir.

    Retorna
    -------
    numpy.ndarray de forma (28, 28) con valores en [0, 1].
    """
    # ── 1. Cargar y convertir a escala de grises ──────────────────────────
    img = Image.open(image_path).convert('L')

    # ── 2. Recortar la prenda (quitar bordes vacíos) ─────────────────────
    # Primero invertimos temporalmente para que PIL detecte el bounding box
    # del contenido no-fondo.
    temp_arr = np.array(img)

    # Detectar si el fondo es claro u oscuro mirando los bordes
    border_pixels = np.concatenate([
        temp_arr[0, :], temp_arr[-1, :],   # filas superior e inferior
        temp_arr[:, 0], temp_arr[:, -1]     # columnas izquierda y derecha
    ])
    bg_brightness = np.mean(border_pixels)
    bg_is_light = bg_brightness > 127

    # Auto-detectar inversión
    should_invert = invert
    if should_invert is None:
        should_invert = bg_is_light  # fondo claro → sí invertir

    # Invertir si necesario (para que la prenda quede CLARA sobre NEGRO)
    if should_invert:
        img = ImageOps.invert(img)

    # ── 3. Aumentar contraste ─────────────────────────────────────────────
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Brightness(img).enhance(1.2)

    # ── 4. Aplicar umbral adaptativo (binarización suave) ────────────────
    # Esto elimina ruido de fondo y hace la imagen más parecida al estilo
    # "silueta" de Fashion MNIST.
    img_array = np.array(img).astype('float32')

    # Umbral: todo lo que sea muy oscuro (ruido de fondo) → negro puro
    threshold = 30  # valor bajo para mantener detalles de la prenda
    img_array[img_array < threshold] = 0

    # ── 5. Recortar al bounding box de la prenda ─────────────────────────
    # Esto centra la prenda y elimina márgenes innecesarios.
    mask = img_array > threshold
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Añadir un pequeño margen (2 px proporcional)
        h, w = img_array.shape
        margin = max(2, int(0.05 * max(rmax - rmin, cmax - cmin)))
        rmin = max(0, rmin - margin)
        rmax = min(h - 1, rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(w - 1, cmax + margin)

        img_array = img_array[rmin:rmax + 1, cmin:cmax + 1]

    # ── 6. Redimensionar a 20×20 y centrar en 28×28 ──────────────────────
    # Fashion MNIST tiene las prendas en ~20×20 px centradas en 28×28.
    cropped_img = Image.fromarray(img_array.astype('uint8'))

    # Mantener la proporción al redimensionar
    cropped_img.thumbnail((20, 20), Image.LANCZOS)

    # Crear imagen final 28×28 con fondo negro y centrar
    final = Image.new('L', (28, 28), color=0)
    offset_x = (28 - cropped_img.width) // 2
    offset_y = (28 - cropped_img.height) // 2
    final.paste(cropped_img, (offset_x, offset_y))

    # ── 7. Normalizar a [0, 1] ───────────────────────────────────────────
    result = np.array(final).astype('float32') / 255.0

    return result


def prepare_real_image_for_mlp(image_path, invert=None):
    """
    Preprocesa una imagen real y la deja lista para el modelo MLP.
    Devuelve array de forma (1, 784).
    """
    img_array = preprocess_real_image(image_path, invert=invert)
    return img_array.reshape(1, 28 * 28)


def prepare_real_image_for_cnn(image_path, invert=None):
    """
    Preprocesa una imagen real y la deja lista para el modelo CNN.
    Devuelve array de forma (1, 28, 28, 1).
    """
    img_array = preprocess_real_image(image_path, invert=invert)
    return img_array.reshape(1, 28, 28, 1)