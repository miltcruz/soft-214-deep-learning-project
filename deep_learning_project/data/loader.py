
import tensorflow as tf
import os
import sqlite3
import pandas as pd

'''
load_mnist_data function:
 - Loads the MNIST dataset using TensorFlow's Keras API
 - Returns the training and test datasets as tuples
 - Normalizes the pixel values to be between 0 and 1
 - Reshapes the data to include a channel dimension for grayscale images
 - The training set consists of 60,000 images and labels
 - The test set consists of 10,000 images and labels
 - The function can be used in the training script to load the MNIST dataset
 - The function can be used in the evaluation script to load the test dataset           
 '''
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # Load MNIST dataset
    x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize pixel values
    x_train = x_train[..., tf.newaxis] # add channel dimension
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test) # return training and test data


def load_ml_data():
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "ml_data.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT feature1, feature2, label FROM training_data", conn)
    conn.close()
    X = df[['feature1', 'feature2']].values
    y = df[['label']].values
    return X, y