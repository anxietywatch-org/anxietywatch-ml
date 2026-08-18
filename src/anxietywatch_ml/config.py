"""
Configuration loader for AnxietyWatch ML.

Loads configuration from YAML file and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file and environment variables.

    Priority: environment variables > YAML config > defaults
    """
    # Load .env file if exists
    load_dotenv()

    # Default configuration
    config = _get_default_config()

    # Load from YAML if provided
    if config_path:
        yaml_path = Path(config_path)
    else:
        # Look for config in standard locations
        for path in [
            Path("configs/base.yaml"),
            Path("config/base.yaml"),
            Path("../configs/base.yaml"),
            Path(__file__).parent.parent.parent / "configs" / "base.yaml",
        ]:
            if path.exists():
                yaml_path = path
                break
        else:
            yaml_path = None

    if yaml_path and yaml_path.exists():
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            config = _deep_merge(config, yaml_config)

    # Override with environment variables
    config = _apply_env_overrides(config)

    return config


def _get_default_config() -> dict:
    """Get default configuration."""
    return {
        "random_seed": 42,
        "window": {
            "size_seconds": 60,
            "stride_seconds": 30,
            "min_samples_per_window": 10,
        },
        "features": {
            "hr_mean": True,
            "hr_std": True,
            "hr_min": True,
            "hr_max": True,
            "hr_slope_bpm_per_min": True,
            "hr_delta_from_baseline": False,
            "hrv_rmssd": True,
            "hrv_sdnn": True,
            "hrv_pnn50": False,
            "movement_magnitude_mean": False,
            "movement_variance_mean": False,
            "skin_temp_mean": True,
            "skin_temp_std": False,
            "quality_good_ratio": True,
            "quality_fair_ratio": True,
            "quality_poor_ratio": True,
            "valid_sample_ratio": True,
            "window_duration_seconds": True,
            "sample_count": True,
            "last_sample_age_seconds": False,
        },
        "model": {
            "type": "logistic_regression",
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 200,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "dummy": {
                "strategy": "prior",
                "constant": 0,
            },
        },
        "training": {
            "test_size": 0.2,
            "val_size": 0.1,
            "group_by": "session",
            "random_state": 42,
        },
        "evaluation": {
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
            ],
            "threshold": 0.5,
        },
        "synthetic": {
            "n_users": 10,
            "n_sessions_per_user": 5,
            "samples_per_session": 300,
            "sampling_rate_hz": 1.0,
            "hr_baseline_mean": 70.0,
            "hr_baseline_std": 8.0,
            "hr_anomaly_shift_bpm": 20.0,
            "ibi_available_probability": 0.3,
            "motion_magnitude_mean": 0.05,
            "motion_magnitude_std": 0.03,
            "skin_temp_mean": 33.5,
            "skin_temp_std": 0.5,
            "anomaly_probability": 0.15,
            "label_noise": 0.05,
        },
        "ground_truth": {
            "window_size_seconds": 60.0,
            "min_samples_per_window": 10,
            "min_hr_ratio": 0.3,
            "support_requested_probability": 0.3,
            "label_noise": 0.05,
            "n_events": 20,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides."""
    env_mappings = {
        "RANDOM_SEED": ("random_seed", int),
        "LOG_LEVEL": ("log_level", str),
        "WINDOW_SIZE_SECONDS": ("window", "size_seconds", int),
        "WINDOW_STRIDE_SECONDS": ("window", "stride_seconds", int),
        "MODEL_TYPE": ("model", "type", str),
    }

    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if isinstance(path, tuple):
                target = config
                for p in path[:-1]:
                    target = target.setdefault(p, {})
                converter = path[-1] if callable(path[-1]) else str
                target[path[-2]] = converter(value)
            else:
                config[path] = value

    return config