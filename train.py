# # def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
# #     history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test))
# #     return history
# def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
#     history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test))
#     return history
from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, callbacks=None):
    if callbacks is None:
        callbacks = []
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)
    return history
