
from deep_learning_project.data.loader import load_mnist_data
from deep_learning_project.models.cnn import create_model
import tensorflow as tf

'''
train function:
 - Loads the MNIST dataset using the loader module
 - Creates a CNN model using the create_model function
 - Compiles the model with Adam optimizer and sparse categorical crossentropy loss
 - Trains the model on the training dataset for 5 epochs
 - Validates the model on the test dataset during training
 - Saves the trained model in HDF5 format
 - This function can be used to train the model on the MNIST dataset
 - The model can be evaluated using the evaluate script after training
 - The model can be used for inference on new MNIST images after training
 - The function can be called from the command line to start training
 - The model can be modified to include more layers or different architectures if needed
 - The model can be saved and loaded for future use without retraining
 - The model can be used as a baseline for further experiments with different architectures or hyperparameters
 - The model can be trained on different datasets or with different hyperparameters if needed
 - The function can be extended to include callbacks for early stopping or model checkpointing
 - The function can be used to log training progress and metrics for monitoring purposes
 - The function can be used to visualize training and validation metrics using TensorBoard
 - The function can be used to generate predictions on the training and test datasets for further analysis
 - The function can be used to save the training results to a file for later use
 - The function can be used to log the training results for monitoring purposes'''
def train():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save("mnist_model.h5")  # Save model to HDF5

if __name__ == "__main__":
    train()
