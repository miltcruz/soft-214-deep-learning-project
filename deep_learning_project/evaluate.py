
import tensorflow as tf
from deep_learning_project.data.loader import load_mnist_data

def evaluate():
    model = tf.keras.models.load_model("mnist_model.h5")
    (_, _), (x_test, y_test) = load_mnist_data()
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate()
