# Sistema de Reconocimiento de Prendas de Vestir — Fashion MNIST

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

Sistema de clasificación de prendas de vestir usando el dataset **Fashion MNIST**. Implementa y compara dos enfoques de Deep Learning:

- **MLP (Red Neuronal Densa):** arquitectura clásica totalmente conectada.
- **CNN (Red Neuronal Convolucional):** arquitectura profunda con capas convolucionales.

El sistema incluye además una **interfaz web** para clasificar imágenes reales tomadas con el dispositivo del usuario.

---

## Clases del Dataset

| Índice | Clase | Descripción |
|--------|-------|-------------|
| 0 | T-shirt/top | Camiseta / top |
| 1 | Trouser | Pantalón |
| 2 | Pullover | Suéter pullover |
| 3 | Dress | Vestido |
| 4 | Coat | Abrigo |
| 5 | Sandal | Sandalia |
| 6 | Shirt | Camisa |
| 7 | Sneaker | Zapatilla deportiva |
| 8 | Bag | Bolso / bolsa |
| 9 | Ankle boot | Botín |

---

## Estructura del Proyecto

```
Fashion-MNIST/
│
├── src/
│   ├── datos_processing.py     # Carga, normalización y preprocesamiento
│   ├── train_mlp.py            # Entrenamiento de la red densa (MLP)
│   ├── train_cnn.py            # Entrenamiento de la red convolucional (CNN)
│   ├── predict.py              # Predicción CLI sobre imágenes reales
│   └── app.py                  # Interfaz web (Gradio)
│
├── models/
│   ├── mlp_model.h5            # Modelo MLP entrenado (generado al correr train_mlp.py)
│   └── cnn_model.h5            # Modelo CNN entrenado (generado al correr train_cnn.py)
│
├── notebooks/
│   ├── eda_preprocesamiento.ipynb     # Análisis exploratorio del dataset
│   ├── test_training_mlp.ipynb        # Notebook de experimentación MLP
│   └── test_training_cnn.ipynb        # Notebook de experimentación CNN
│
├── requirements.txt
└── README.md
```

---

## Instalación y Configuración

### Requisitos

- **Python 3.10 – 3.12** (recomendado: Python 3.12)
  > ⚠️ TensorFlow no es compatible con Python 3.13+. Usa `python --version` para verificar.

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/valexq/Fashion-MNIST.git
cd Fashion-MNIST

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Activar en Windows:
venv\Scripts\activate

# Activar en Linux/macOS:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Flujo de Ejecución

### Paso 1 — Análisis Exploratorio (opcional)

Abre el notebook de EDA para explorar el dataset:

```bash
jupyter notebook notebooks/eda_preprocesamiento.ipynb
```

### Paso 2 — Explorar y entrenar el modelo MLP

```bash
# Opción A: Notebook interactivo (recomendado para exploración)
jupyter notebook notebooks/test_training_mlp.ipynb

# Opción B: Script directo
python src/train_mlp.py
```

El modelo se guardará en `models/mlp_model.h5`.

**Resultados esperados MLP:**
- Accuracy en test: **~88–89%**
- Arquitectura: `784 → Dense(128) → Dropout(0.2) → Dense(64) → Dense(10)`

### Paso 3 — Explorar y entrenar el modelo CNN

```bash
# Opción A: Notebook interactivo
jupyter notebook notebooks/test_training_cnn.ipynb

# Opción B: Script directo
python src/train_cnn.py
```

El modelo se guardará en `models/cnn_model.h5`.

**Resultados esperados CNN:**
- Accuracy en test: **~91–93%**
- Arquitectura: 2 bloques ConvNet + Clasificador denso

### Paso 4 — Clasificar imágenes reales (CLI)

```bash
# Con CNN (recomendado, mayor precisión)
python src/predict.py --image ruta/a/tu/imagen.jpg --model cnn

# Con MLP
python src/predict.py --image ruta/a/tu/imagen.jpg --model mlp

# Si el fondo de la imagen es oscuro, agrega --invert
python src/predict.py --image imagen.jpg --model cnn --invert
```

**Salida esperada:**
```
Cargando modelo 'cnn' desde: .../models/cnn_model.h5

=============================================
  Predicción: Sneaker
  Confianza:  94.3%
=============================================

Top-3 clases:
  1. Sneaker         [████████████████░░░░]  94.3%
  2. Ankle boot      [█░░░░░░░░░░░░░░░░░░░]   4.1%
  3. Sandal          [░░░░░░░░░░░░░░░░░░░░]   1.2%
```

### Paso 5 — Interfaz Web (Gradio)

```bash
python src/app.py
```

Abrir en el navegador: **http://localhost:7860**

La interfaz permite:
- 📷 Subir una imagen de prenda
- 🤖 Elegir el modelo (MLP o CNN)
- 🔄 Activar inversión de colores si es necesario
- 📊 Ver las probabilidades por clase

---

## Arquitecturas de los Modelos

### MLP (Red Densa)

```
Input (784) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(10, Softmax)
```

| Parámetro | Valor |
|-----------|-------|
| Optimizer | Adam |
| Loss | sparse_categorical_crossentropy |
| Batch size | 32 |
| Max épocas | 10 |
| EarlyStopping | patience=3, monitor=val_loss |

### CNN (Red Convolucional)

```
Input (28×28×1)
→ Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
→ Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
→ Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(10, Softmax)
```

| Parámetro | Valor |
|-----------|-------|
| Optimizer | Adam |
| Loss | sparse_categorical_crossentropy |
| Batch size | 64 |
| Max épocas | 20 |
| EarlyStopping | patience=3, monitor=val_loss |

---

## Comparativa de Resultados

| Modelo | Accuracy en Test | Parámetros | Tiempo de entrenamiento |
|--------|-----------------|------------|-------------------------|
| MLP    | ~88–89%          | ~120 K     | ~1–2 min (CPU) |
| CNN    | ~91–93%          | ~450 K     | ~5–10 min (CPU) |

> La CNN supera al MLP porque aprovecha la **estructura espacial** de las imágenes (bordes, texturas, formas), algo que la red densa no puede hacer al tratar cada píxel de forma independiente.

---

## Consejos para Imágenes Reales

- ✅ Usa **buena iluminación** (luz natural o artificial uniforme).
- ✅ Usa **fondo claro** (blanco o gris preferiblemente).
- ✅ Muestra **una sola prenda** por imagen.
- ✅ Centra la prenda en el encuadre.
- ⚙️ Si la predicción es incorrecta, prueba con `--invert` (invierte los colores).
- ⚙️ La CNN suele ser más robusta ante variaciones de iluminación y ángulo.

---

## Módulo de Preprocesamiento (`datos_processing.py`)

| Función | Descripción |
|---------|-------------|
| `load_fashion_mnist()` | Carga el dataset original desde TensorFlow |
| `normalize_images(x_train, x_test)` | Normaliza píxeles al rango [0, 1] |
| `prepare_for_mlp(images)` | Reshape a `(N, 784)` para red densa |
| `prepare_for_cnn(images)` | Reshape a `(N, 28, 28, 1)` para CNN |
| `load_and_prepare_all()` | Carga + normaliza + prepara todo en un dict |
| `preprocess_real_image(path, invert)` | Convierte imagen real a formato Fashion MNIST |
| `prepare_real_image_for_mlp(path, invert)` | Imagen real → formato MLP |
| `prepare_real_image_for_cnn(path, invert)` | Imagen real → formato CNN |

---

## Entregables del Proyecto

- [x] Código fuente del sistema (`src/`)
- [x] Notebook de análisis exploratorio (`eda_preprocesamiento.ipynb`)
- [x] Notebook de prueba MLP (`test_training_mlp.ipynb`)
- [x] Notebook de prueba CNN (`test_training_cnn.ipynb`)
- [x] Interfaz web (`app.py`)
- [x] Script de predicción CLI (`predict.py`)
- [ ] Modelos entrenados (`models/`) — se generan al ejecutar los scripts
- [ ] Informe final
- [ ] Video de presentación
