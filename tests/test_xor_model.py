import unittest
import numpy as np
from deep_learning_project.data.loader import load_ml_data
from deep_learning_project.models.xor_model import XORModel

class TestXORModel(unittest.TestCase):
    def test_training(self):
        X, y = load_ml_data()
        model = XORModel()
        model.train(X, y, epochs=10)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (4, 1))

if __name__ == "__main__":
    unittest.main()