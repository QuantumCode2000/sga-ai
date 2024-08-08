from sklearn.preprocessing import MinMaxScaler

def get_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler

def scale_data(data, scaler):
    original_shape = data.shape
    data = data.reshape(-1, original_shape[-1])
    data = scaler.transform(data)
    return data.reshape(original_shape)
