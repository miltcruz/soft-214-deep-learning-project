
from deep_learning_project.data.loader import load_ml_data
from deep_learning_project.models.xor_model import XORModel


def main():
    X, y = load_ml_data()

    model = XORModel()
    print("Training the XOR model...\n")
    model.train(X, y, epochs=1000, verbose=1)

    loss, accuracy = model.evaluate(X, y)
    print(f"\nEvaluation results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")

    print("Predictions:")
    predictions = model.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} → Predicted: {predictions[i][0]:.4f} → Rounded: {round(predictions[i][0])}")

if __name__ == "__main__":
    main()
