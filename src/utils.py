"""
Contains various utility functions for PyTorch model training and saving.
"""
from flax import serialization
from pathlib import Path
import os

def save_model(model_params,
               target_dir: str,
               model_name: str):
    """Saves JAX model parameters to a file in a target directory.

        Args:
        model_params: JAX model parameters to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
            either ".pkl" or ".params" as the file extension.

        Example usage:
        save_model(model_params, "models", "model_name.pkl")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pkl") or model_name.endswith(".params"), "model_name should end with '.pkl' or '.params'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    with open(model_save_path, "wb") as f:
        serialization.to_bytes(model_params, f)

