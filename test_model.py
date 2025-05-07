import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Prediction list should not be empty."
    assert len(preds) == len(y_test), "Prediction and true labels must match in length."

def test_predictions_value_range():
    preds, _ = train_and_predict()
    unique_values = set(preds)
    expected_values = {0, 1, 2}
    assert unique_values.issubset(expected_values), f"Predictions contain unexpected values: {unique_values}. Expected values are {expected_values}."

def test_model_accuracy():
    accuracy = get_accuracy()
    assert accuracy >= 0.7, f"Model accuracy is below 70%. Current accuracy: {accuracy}."