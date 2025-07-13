
from deep_learning_project.data.loader import load_mnist_data
from deep_learning_project.models.cnn import create_model
import tensorflow as tf

def train():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save("mnist_model.h5")

if __name__ == "__main__":
    train()
