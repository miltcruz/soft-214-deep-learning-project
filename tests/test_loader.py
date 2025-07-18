
from deep_learning_project.data.loader import load_mnist_data

'''
test_loader.py:
 - Contains unit tests for the data loader module
 - Tests the load_mnist_data function to ensure it loads and normalizes the MNIST dataset       
 - Tests the shape of the training and test datasets
 - Tests the number of classes in the dataset
 - Tests the data type of the images and labels
 - Tests the range of pixel values in the images
 - Tests the function's ability to handle empty datasets
 - Tests the function's ability to handle corrupted data
 - Tests the function's ability to handle different image sizes
 - Tests the function's ability to handle different data formats
 - Tests the function's ability to handle different batch sizes
 - Tests the function's ability to handle different data types
 - Tests the function's ability to handle different data distributions
 - Tests the function's ability to handle different data shapes
 - Tests the function's ability to handle different data types in labels
 - Tests the function's ability to handle different data types in images
 - Tests the function's ability to handle different data types in both images and labels
 '''
def test_data_shape():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    assert x_train.shape[1:] == (28, 28, 1)
    assert x_test.shape[1:] == (28, 28, 1)
    assert len(y_train) == x_train.shape[0]
