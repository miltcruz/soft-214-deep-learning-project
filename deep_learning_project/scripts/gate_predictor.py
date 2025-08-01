"""
gate_predictor.py

Use a freshly initialized XORModel to perform inference on logic gate inputs.
"""

# Import the data loading function from our custom data module
from deep_learning_project.data.loader import load_ml_data
# Import our custom XOR neural network model
from deep_learning_project.models.xor_model import XORModel
# Import NumPy for numerical operations and array handling
import numpy as np

def predict_output(model, x1, x2):
    """
    Make a prediction using the trained model for given inputs.
    
    Args:
        model: The trained XOR neural network model
        x1: First input value (0 or 1)
        x2: Second input value (0 or 1)
    
    Returns:
        tuple: (binary_prediction, raw_prediction)
    """
    # Create a 2D NumPy array with the input values
    # Shape is (1, 2) meaning 1 sample with 2 features
    input_data = np.array([[x1, x2]])
    
    # Use the model to make a prediction
    # predict() returns a 2D array, so we use [0][0] to get the scalar value
    prediction = model.predict(input_data)[0][0]
    
    # Convert the raw prediction (0.0 to 1.0) to binary (0 or 1)
    # If prediction > 0.5, classify as 1, otherwise 0
    binary_prediction = int(prediction > 0.5)
    
    # Return both the binary result and the raw probability
    return binary_prediction, prediction

def main():
    """
    Main function that demonstrates how to use the XOR model for prediction.
    This function trains a new model and tests it on all possible XOR inputs.
    """
    # Load the training data from our database
    # X contains input pairs like [0,0], [0,1], [1,0], [1,1]
    # y contains expected outputs like [0], [1], [1], [0] for XOR logic
    X, y = load_ml_data()

    # Create a new instance of our XOR neural network model
    # This model is not trained yet - it has random weights
    model = XORModel()
    
    # Train the model using the loaded data
    # epochs=1000: Run 1000 training iterations to learn the XOR pattern
    # verbose=0: Don't print training progress (silent mode)
    model.train(X, y, epochs=1000, verbose=0)

    # Define all possible input combinations for XOR logic gate
    # XOR returns 1 when inputs are different, 0 when they're the same
    test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected_outputs = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0
    }
    # Print header for our results
    print("Gate Predictor Output:")
    
    # Loop through each test case and make predictions
    for x1, x2 in test_cases:
        expected = expected_outputs[(x1, x2)]
        """Run a prediction using the XOR model."""
        # Get both the binary prediction (0 or 1) and raw probability
        binary, raw = predict_output(model, x1, x2)
        is_correct = "✅" if binary == expected else "❌"
        # Display the input values, predicted output, and raw probability
        # Raw probability shows how confident the model is (closer to 0 or 1 = more confident)
        print(f"Input: ({x1}, {x2}) → Predicted: {binary} (Raw: {raw:.4f}) | Expected: {expected} {is_correct}")


# This conditional ensures main() only runs when this file is executed directly
# It won't run if this file is imported as a module into another Python file
if __name__ == "__main__":
    # Call the main function to start the program
    main()
