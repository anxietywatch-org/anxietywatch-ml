"""
Serialization utilities for AnxietyWatch ML artifacts.

WARNING: pickle can execute arbitrary code while loading.
Only load artifacts produced by this project from trusted storage.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class PickleArtifactSerializer:
    """Small versioned serializer for trusted AnxietyWatch ML artifacts."""

    FORMAT_VERSION = 2

    def save(
        self,
        artifact: Any,
        path: Path | str,
        *,
        artifact_type: str,
    ) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "artifact": artifact,
            "artifact_type": artifact_type,
            "format_version": self.FORMAT_VERSION,
            "serialized_by": "anxietywatch-ml",
        }

        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(
        self,
        path: Path | str,
        *,
        expected_type: type[T],
        expected_artifact_type: str,
    ) -> T:
        input_path = Path(path)

        with input_path.open("rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("Invalid artifact format: expected a metadata dictionary")

        if payload.get("serialized_by") != "anxietywatch-ml":
            raise ValueError("Invalid artifact format: unexpected serializer")

        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported artifact format version: "
                f"{payload.get('format_version')!r}"
            )

        if payload.get("artifact_type") != expected_artifact_type:
            raise ValueError(
                "Unexpected artifact type: "
                f"{payload.get('artifact_type')!r}; "
                f"expected {expected_artifact_type!r}"
            )

        artifact = payload.get("artifact")
        if not isinstance(artifact, expected_type):
            raise ValueError(
                f"Loaded artifact has type {type(artifact)!r}; "
                f"expected {expected_type!r}"
            )

        return artifact


def save_artifact(
    artifact: Any,
    path: Path | str,
    *,
    artifact_type: str,
) -> None:
    PickleArtifactSerializer().save(
        artifact,
        path,
        artifact_type=artifact_type,
    )


def load_artifact(
    path: Path | str,
    *,
    expected_type: type[T],
    expected_artifact_type: str,
) -> T:
    return PickleArtifactSerializer().load(
        path,
        expected_type=expected_type,
        expected_artifact_type=expected_artifact_type,
    )


# Backwards-compatible helpers for standalone baseline-model tests.
def save_model(model: Any, path: Path | str) -> None:
    if not getattr(model, "_is_fitted", False):
        raise RuntimeError("Cannot save unfitted model")

    save_artifact(
        model,
        path,
        artifact_type="baseline_model",
    )


def load_model(path: Path | str, model_type: type[T]) -> T:
    model = load_artifact(
        path,
        expected_type=model_type,
        expected_artifact_type="baseline_model",
    )

    if not getattr(model, "_is_fitted", False):
        raise RuntimeError("Loaded model is not fitted")

    return model
