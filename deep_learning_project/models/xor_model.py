import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class XORModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Hidden layer: more neurons + tanh improves XOR learning
        model.add(Dense(4, input_dim=2, activation='tanh'))
        # Output layer: sigmoid for binary classification
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )
        return model

    def train(self, X, y, epochs=3000, verbose=1):
        return self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)

    def predict(self, X):
        return self.model.predict(X)