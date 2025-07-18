
import tensorflow as tf
from deep_learning_project.data.loader import load_mnist_data

'''
evaluate function:
 - Loads the trained CNN model from HDF5 format
 - Loads the MNIST test dataset using the loader module
 - Evaluates the model on the test dataset
 - Prints the test accuracy of the model
 - This function can be used to assess the performance of the model after training
 - The model can be used for inference on new MNIST images after evaluation
 - The function can be called from the command line to evaluate the model
 - The model can be modified to include more layers or different architectures if needed
 - The model can be saved and loaded for future use without retraining
 - The model can be used as a baseline for further experiments with different architectures or hyperparameters
 - The model can be evaluated on different datasets or with different metrics if needed
 - The function can be used to compare the performance of different models or architectures
 - The function can be extended to include more evaluation metrics or visualizations
 - The function can be used to generate predictions on the test dataset for further analysis
 - The function can be used to save the evaluation results to a file for later use
 - The function can be used to log the evaluation results for monitoring purposes
 '''
def evaluate():
    model = tf.keras.models.load_model("mnist_model.h5") # Load HDF5 model
    (_, _), (x_test, y_test) = load_mnist_data()
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate()
