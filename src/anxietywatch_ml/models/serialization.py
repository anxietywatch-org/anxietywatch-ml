"""
Model serialization utilities for AnxietyWatch ML.

This module isolates pickle-based serialization behind a clean interface.
WARNING: Never load pickle files from untrusted sources.
Pickle can execute arbitrary code during deserialization.
"""

import pickle
from pathlib import Path
from typing import Any, Protocol

# Use Protocol with duck typing to avoid circular imports
class BaselineModelProtocol(Protocol):
    """Protocol for baseline models - avoids circular import."""
    _is_fitted: bool
    pipeline: Any
    _feature_names: list[str]
    config: Any


class ModelSerializer(Protocol):
    """Protocol for model serialization."""

    def save(self, model: BaselineModelProtocol, path: Path) -> None:
        """Save model to path."""
        ...

    def load(self, path: Path, model_type: type) -> BaselineModelProtocol:
        """Load model from path."""
        ...


class PickleModelSerializer:
    """
    Pickle-based model serializer.

    SECURITY WARNING: Never load pickle files from untrusted sources.
    Pickle can execute arbitrary code during deserialization.
    Only use with models saved by this application.
    """

    def save(self, model: BaselineModelProtocol, path: Path) -> None:
        """Save model to path using pickle."""
        if not model._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata for validation
        data = {
            "model": model,
            "format_version": 1,
            "serialized_by": "anxietywatch-ml",
        }

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path, model_type: type) -> BaselineModelProtocol:
        """Load model from path using pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Validate format
        if not isinstance(data, dict):
            raise ValueError("Invalid model format: expected dict")

        if "model" not in data:
            raise ValueError("Invalid model format: missing 'model' key")

        model = data["model"]

        if not isinstance(model, model_type):
            raise ValueError(
                f"Loaded model type {type(model)} does not match expected {model_type}"
            )

        if not model._is_fitted:
            raise RuntimeError("Loaded model is not fitted")

        return model


# Default serializer instance
default_serializer = PickleModelSerializer()


def save_model(model: BaselineModelProtocol, path: Path | str) -> None:
    """
    Save a fitted model to disk.

    Args:
        model: Fitted BaselineModel instance
        path: Output path (will be created if needed)

    Raises:
        RuntimeError: If model is not fitted
        OSError: If path cannot be written
    """
    default_serializer.save(model, Path(path))


def load_model(path: Path | str, model_type: type) -> BaselineModelProtocol:
    """
    Load a fitted model from disk.

    SECURITY WARNING: Only load models saved by this application.
    Never load pickle files from untrusted sources.

    Args:
        path: Path to model file
        model_type: Expected model type

    Returns:
        Fitted model instance

    Raises:
        ValueError: If model format is invalid or type mismatch
        RuntimeError: If loaded model is not fitted
        OSError: If file cannot be read
    """
    return default_serializer.load(Path(path), model_type)