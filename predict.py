import numpy as np

def predict(model, X_test, scaler_y, name):
    if X_test.size == 0:
        raise ValueError("X_test is empty. Ensure that the dataset is prepared correctly.")
    
    print(f"Shape of X_test: {X_test.shape}")
    
    scaled_predictions = model.predict(X_test)
    predicted_nacimientos = scaler_y.inverse_transform(scaled_predictions)
    print(f"{name} - Nacimientos predichos para el próximo mes: {predicted_nacimientos.flatten()}")
    return predicted_nacimientos.flatten()

def predict_all(models, X_test, scaler_y):
    predictions = {}
    for name, model in models.items():
        predictions[name] = predict(model, X_test, scaler_y, name)
    return predictions
