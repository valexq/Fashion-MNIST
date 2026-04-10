from datos_preprocessing import load_and_prepare_all

data = load_and_prepare_all()

x_train = data["x_train_mlp"]
y_train = data["y_train"]
x_test = data["x_test_mlp"]
y_test = data["y_test"]
