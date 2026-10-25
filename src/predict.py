"""Inference entry point."""

def predict(model, features):
    """Predict labels using an already trained model."""
    if not hasattr(model, "predict"):
        raise TypeError("model must implement predict().")

    return model.predict(features)
