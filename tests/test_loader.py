
from deep_learning_project.data.loader import load_mnist_data

def test_data_shape():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    assert x_train.shape[1:] == (28, 28, 1)
    assert x_test.shape[1:] == (28, 28, 1)
    assert len(y_train) == x_train.shape[0]
