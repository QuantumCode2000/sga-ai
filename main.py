# # import matplotlib.pyplot as plt
# # from models.cnn_simple import build_cnn_simple
# # from models.cnn_with_pooling import build_cnn_with_pooling
# # from models.mlp import build_mlp
# # from models.lstm import build_lstm
# # from train import train_model
# # from predict import predict_all
# # from utils.data_preparation import load_data, preprocess_data
# # from utils.scaler import get_scaler, scale_data
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # # Nombre de la columna objetivo
# # TARGET_COLUMN = 'Nacimientos'

# # # Cargar y preparar los datos de entrenamiento y prueba
# # train_data = load_data('train_data.csv')
# # test_data = load_data('test_data.csv')

# # X_train, y_train = preprocess_data(train_data, TARGET_COLUMN)
# # X_test, y_test = preprocess_data(test_data, TARGET_COLUMN)

# # # Obtener el escalador y escalar los datos
# # scaler_X = get_scaler(X_train.reshape(-1, X_train.shape[-1]))
# # scaler_y = get_scaler(y_train.reshape(-1, 1))

# # X_train = scale_data(X_train, scaler_X)
# # X_test = scale_data(X_test, scaler_X)

# # y_train = scaler_y.transform(y_train.reshape(-1, 1)).reshape(-1)
# # y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

# # # Construir los modelos
# # cnn_simple = build_cnn_simple((X_train.shape[1], X_train.shape[2]))
# # cnn_with_pooling = build_cnn_with_pooling((X_train.shape[1], X_train.shape[2]))
# # mlp = build_mlp((X_train.shape[1] * X_train.shape[2],))
# # lstm = build_lstm((X_train.shape[1], X_train.shape[2]))

# # # Entrenar los modelos
# # epochs = 100
# # batch_size = 32

# # print("Training CNN Simple...")
# # history_cnn_simple = train_model(cnn_simple, X_train, y_train, X_test, y_test, epochs, batch_size)

# # print("Training CNN with Pooling...")
# # history_cnn_with_pooling = train_model(cnn_with_pooling, X_train, y_train, X_test, y_test, epochs, batch_size)

# # print("Training MLP...")
# # history_mlp = train_model(mlp, X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, epochs, batch_size)

# # print("Training LSTM...")
# # history_lstm = train_model(lstm, X_train, y_train, X_test, y_test, epochs, batch_size)

# # # Hacer predicciones
# # models = {
# #     "CNN Simple": cnn_simple,
# #     "CNN with Pooling": cnn_with_pooling,
# #     "MLP": mlp,
# #     "LSTM": lstm
# # }

# # predictions = predict_all(models, X_test, scaler_y)

# # # Evaluar modelos
# # def evaluate_model(model, X_test, y_test, scaler_X, scaler_y, name):
# #     y_pred = model.predict(X_test)
# #     y_pred = scaler_y.inverse_transform(y_pred).flatten()
# #     y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# #     mae = mean_absolute_error(y_test_inv, y_pred)
# #     mse = mean_squared_error(y_test_inv, y_pred)
# #     r2 = r2_score(y_test_inv, y_pred)

# #     print(f"{name} - MAE: {mae}, MSE: {mse}, R2: {r2}")
    
# #     plt.figure(figsize=(10, 5))
# #     plt.plot(y_test_inv, label='True Values')
# #     plt.plot(y_pred, label='Predicted Values')
# #     plt.title(f'{name} Predictions')
# #     plt.xlabel('Samples')
# #     plt.ylabel('Nacimientos')
# #     plt.legend()
# #     plt.show()

# # for name, model in models.items():
# #     evaluate_model(model, X_test, y_test, scaler_X, scaler_y, name)
# import matplotlib.pyplot as plt
# from models.cnn_simple import build_cnn_simple
# from models.cnn_with_pooling import build_cnn_with_pooling
# from models.mlp import build_mlp
# from models.lstm import build_lstm
# from train import train_model
# from predict import predict_all
# from utils.data_preparation import load_data, preprocess_data
# from utils.scaler import get_scaler, scale_data
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Nombre de la columna objetivo
# TARGET_COLUMN = 'Nacimientos'

# # Cargar y preparar los datos de entrenamiento y prueba
# train_data = load_data('train_data.csv')
# test_data = load_data('test_data.csv')

# X_train, y_train = preprocess_data(train_data, TARGET_COLUMN)
# X_test, y_test = preprocess_data(test_data, TARGET_COLUMN)

# # Obtener el escalador y escalar los datos
# scaler_X = get_scaler(X_train.reshape(-1, X_train.shape[-1]))
# scaler_y = get_scaler(y_train.reshape(-1, 1))

# X_train = scale_data(X_train, scaler_X)
# X_test = scale_data(X_test, scaler_X)

# y_train = scaler_y.transform(y_train.reshape(-1, 1)).reshape(-1)
# y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

# # Construir los modelos
# cnn_simple = build_cnn_simple((X_train.shape[1], X_train.shape[2]))
# cnn_with_pooling = build_cnn_with_pooling((X_train.shape[1], X_train.shape[2]))
# mlp = build_mlp((X_train.shape[1] * X_train.shape[2],))
# lstm = build_lstm((X_train.shape[1], X_train.shape[2]))

# # Entrenar los modelos
# epochs = 100
# batch_size = 32

# print("Entrenando CNN Simple...")
# history_cnn_simple = train_model(cnn_simple, X_train, y_train, X_test, y_test, epochs, batch_size)

# print("Entrenando CNN con Pooling...")
# history_cnn_with_pooling = train_model(cnn_with_pooling, X_train, y_train, X_test, y_test, epochs, batch_size)

# print("Entrenando MLP...")
# history_mlp = train_model(mlp, X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, epochs, batch_size)

# print("Entrenando LSTM...")
# history_lstm = train_model(lstm, X_train, y_train, X_test, y_test, epochs, batch_size)

# # Hacer predicciones
# models = {
#     "CNN Simple": cnn_simple,
#     "CNN con Pooling": cnn_with_pooling,
#     "MLP": mlp,
#     "LSTM": lstm
# }

# predictions = predict_all(models, X_test, scaler_y)

# # Evaluar modelos
# def evaluate_model(model, X_test, y_test, scaler_X, scaler_y, name):
#     y_pred = model.predict(X_test)
#     y_pred = scaler_y.inverse_transform(y_pred).flatten()
#     y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

#     mae = mean_absolute_error(y_test_inv, y_pred)
#     mse = mean_squared_error(y_test_inv, y_pred)
#     r2 = r2_score(y_test_inv, y_pred)

#     print(f"{name} - MAE: {mae}, MSE: {mse}, R2: {r2}")
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(y_test_inv, label='Valores Verdaderos')
#     plt.plot(y_pred, label='Valores Predichos')
#     plt.title(f'Predicciones de {name}')
#     plt.xlabel('Muestras')
#     plt.ylabel('Nacimientos')
#     plt.legend()
#     plt.show()

# for name, model in models.items():
#     evaluate_model(model, X_test, y_test, scaler_X, scaler_y, name)

import matplotlib.pyplot as plt
from models.cnn_simple import build_cnn_simple
from models.cnn_with_pooling import build_cnn_with_pooling
from models.mlp import build_mlp
from models.lstm import build_lstm
from train import train_model
from predict import predict_all
from utils.data_preparation import load_data, preprocess_data
from utils.scaler import get_scaler, scale_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Nombre de la columna objetivo
TARGET_COLUMN = 'Nacimientos'

# Cargar y preparar los datos de entrenamiento y prueba
train_data = load_data('train_data.csv')
test_data = load_data('test_data.csv')

X_train, y_train = preprocess_data(train_data, TARGET_COLUMN)
X_test, y_test = preprocess_data(test_data, TARGET_COLUMN)

# Obtener el escalador y escalar los datos
scaler_X = get_scaler(X_train.reshape(-1, X_train.shape[-1]))
scaler_y = get_scaler(y_train.reshape(-1, 1))

X_train = scale_data(X_train, scaler_X)
X_test = scale_data(X_test, scaler_X)

y_train = scaler_y.transform(y_train.reshape(-1, 1)).reshape(-1)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

# Construir los modelos
cnn_simple = build_cnn_simple((X_train.shape[1], X_train.shape[2]))
cnn_with_pooling = build_cnn_with_pooling((X_train.shape[1], X_train.shape[2]))
mlp = build_mlp((X_train.shape[1] * X_train.shape[2],))
lstm = build_lstm((X_train.shape[1], X_train.shape[2]))

# Entrenar los modelos
epochs = 100
batch_size = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

print("Training CNN Simple...")
history_cnn_simple = train_model(cnn_simple, X_train, y_train, X_test, y_test, epochs, batch_size, [early_stopping])

print("Training CNN with Pooling...")
history_cnn_with_pooling = train_model(cnn_with_pooling, X_train, y_train, X_test, y_test, epochs, batch_size, [early_stopping])

print("Training MLP...")
history_mlp = train_model(mlp, X_train.reshape(X_train.shape[0], -1), y_train, X_test.reshape(X_test.shape[0], -1), y_test, epochs, batch_size, [early_stopping])

print("Training LSTM...")
history_lstm = train_model(lstm, X_train, y_train, X_test, y_test, epochs, batch_size, [early_stopping])

# Hacer predicciones
models = {
    "CNN Simple": cnn_simple,
    "CNN with Pooling": cnn_with_pooling,
    "MLP": mlp,
    "LSTM": lstm
}

predictions = predict_all(models, X_test, scaler_y)

# Evaluar modelos
def evaluate_model(model, X_test, y_test, scaler_X, scaler_y, name):
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_inv, y_pred)
    mse = mean_squared_error(y_test_inv, y_pred)
    r2 = r2_score(y_test_inv, y_pred)

    print(f"{name} - MAE: {mae}, MSE: {mse}, R2: {r2}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Valores Verdaderos')
    plt.plot(y_pred, label='Valores Predichos')
    plt.title(f'Predicciones de {name}')
    plt.xlabel('Muestras')
    plt.ylabel('Nacimientos')
    plt.legend()
    plt.show()

for name, model in models.items():
    evaluate_model(model, X_test, y_test, scaler_X, scaler_y, name)
