import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class XORModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(2, input_dim=2, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model

    def train(self, X, y, epochs=1000, verbose=1):
        return self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)

    def predict(self, X):
        return self.model.predict(X)