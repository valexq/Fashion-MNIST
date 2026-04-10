# Sistema de reconocimiento de prendas de vestir con Fashion MNIST

Este proyecto implementa un sistema de clasificación de prendas de vestir usando el dataset **Fashion MNIST**. Se desarrollan dos enfoques de aprendizaje automático:

- Una red neuronal básica (**MLP**).
- Una red neuronal convolucional (**CNN**).

Además, el sistema permite probar imágenes reales tomadas desde el dispositivo del usuario, aplicando un preprocesamiento compatible con el formato del dataset.

## Objetivo

Construir un sistema capaz de clasificar imágenes de prendas en una de las 10 clases de Fashion MNIST y comparar el desempeño entre un modelo MLP y un modelo CNN.

## Clases del dataset

Las clases utilizadas son:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Estructura del proyecto

```bash
Fashion-MNIST/
│
├── src/
│   ├── datos_preprocessing.py
│   ├── train_mlp.py
│   ├── train_cnn.py
│   ├── predict.py
│   └── app.py
│
├── notebooks/
│   └── eda_preprocesamiento.ipynb
│
├── requirements.txt
└── README.md
```

## Instalación

1. Crear y activar un entorno virtual.
2. Instalar las dependencias del proyecto:

```bash
python -m pip install -r requirements.txt
```

> Nota: para instalar TensorFlow se recomienda usar **Python 3.12**, ya que versiones más recientes como Python 3.14 pueden no ser compatibles.

## Flujo del sistema

### 1. Preprocesamiento

El archivo `fashion_preprocessing.py` se encarga de:

- Cargar el dataset Fashion MNIST desde TensorFlow.
- Normalizar las imágenes a valores entre 0 y 1.
- Preparar los datos para MLP en formato `(N, 784)`.
- Preparar los datos para CNN en formato `(N, 28, 28, 1)`.
- Preprocesar imágenes reales tomadas desde el dispositivo.

### 2. Entrenamiento del modelo MLP

Ejecutar:

```bash
python src/train_mlp.py
```

Este script:
- Carga los datos preprocesados.
- Entrena una red neuronal densa.
- Evalúa su desempeño.
- Guarda el modelo en `models/mlp_model.h5`.

### 3. Entrenamiento del modelo CNN

Ejecutar:

```bash
python src/train_cnn.py
```

Este script:
- Carga los datos preprocesados.
- Entrena una red convolucional.
- Evalúa su desempeño.
- Guarda el modelo en `models/cnn_model.h5`.

### 4. Predicción con imágenes reales

El archivo `predict.py` permite cargar un modelo ya entrenado y realizar predicciones sobre imágenes reales.

### 5. Interfaz del sistema

El archivo `app.py` corresponde a la interfaz del sistema, donde el usuario puede cargar una imagen y seleccionar el modelo con el cual desea hacer la clasificación.

## Funciones principales

### `fashion_preprocessing.py`

- `load_fashion_mnist()`
- `normalize_images(x_train, x_test)`
- `prepare_for_mlp(images)`
- `prepare_for_cnn(images)`
- `load_and_prepare_all()`
- `preprocess_real_image(image_path, invert=False)`
- `prepare_real_image_for_mlp(image_path, invert=False)`
- `prepare_real_image_for_cnn(image_path, invert=False)`

## Pruebas con imágenes reales

Para imágenes reales se recomienda:

- Tomar la foto con buena iluminación.
- Usar fondo simple.
- Mostrar una sola prenda por imagen.
- Probar también con `invert=True` si el contraste no coincide con el dataset.

## Entregables

- Código fuente del sistema.
- Notebook de análisis exploratorio.
- Modelos entrenados.
- Informe final.
- Video de presentación.
