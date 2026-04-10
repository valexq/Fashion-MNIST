import numpy as np
from PIL import Image, ImageOps
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

# Función que carga el dataset
def load_fashion_mnist():
    return fashion_mnist.load_data()

# Función que normaliza la escala de píxeles al rango
def normalize_images(x_train, x_test):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test

# Función que convierte cada imagen 28×28 en un vector de 784 para la red densa
def prepare_for_mlp(images):
    return images.reshape(images.shape[0], 28 * 28)

# Función que convierte a (N, 28, 28, 1)
def prepare_for_cnn(images):
    return images.reshape(images.shape[0], 28, 28, 1)

# Función que se usa en train_mlp y train_cnn para entrenar el modelo
# Carga, normaliza y devuelve un diccionario
def load_and_prepare_all():
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    x_train, x_test = normalize_images(x_train, x_test)

    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'x_train_mlp': prepare_for_mlp(x_train),
        'x_test_mlp': prepare_for_mlp(x_test),
        'x_train_cnn': prepare_for_cnn(x_train),
        'x_test_cnn': prepare_for_cnn(x_test),
        'class_names': class_names
    }
    return data


# Función que transforma una foto real a formato Fashion MNIST
def preprocess_real_image(image_path, invert=False):
    img = Image.open(image_path).convert('L')
    img = ImageOps.fit(img, (28, 28))

    if invert:
        img = ImageOps.invert(img)

    img_array = np.array(img).astype('float32') / 255.0
    return img_array

# Se usa en el predict.py y en la interfaz
# Función que entrega el formato para el mlp
def prepare_real_image_for_mlp(image_path, invert=False):
    img_array = preprocess_real_image(image_path, invert=invert)
    return img_array.reshape(1, 28 * 28)

# Se usa en el predict.py y en la interfaz
# Función que entrega el formato para el cnn
def prepare_real_image_for_cnn(image_path, invert=False):
    img_array = preprocess_real_image(image_path, invert=invert)
    return img_array.reshape(1, 28, 28, 1)