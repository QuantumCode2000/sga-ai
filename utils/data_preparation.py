import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    features = data.drop(columns=['Fecha', target_column])
    target = data[target_column]
    return features.values.reshape((features.shape[0], features.shape[1], 1)), target.values
