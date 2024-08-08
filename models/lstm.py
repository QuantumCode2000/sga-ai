# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Input

# def build_lstm(input_shape):
#     model = Sequential([
#         LSTM(50, input_shape=input_shape),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
#     return model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model
