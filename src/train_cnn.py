from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from datos_preprocessing import load_and_prepare_all

data = load_and_prepare_all()

x_train = data["x_train_cnn"]
y_train = data["y_train"]
x_test = data["x_test_cnn"]
y_test = data["y_test"]
