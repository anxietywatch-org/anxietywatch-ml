"""
CLI entry point for AnxietyWatch ML.

Provides commands for training, prediction, and data generation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler

from anxietywatch_ml.config import load_config
from anxietywatch_ml.data.synthetic import create_generator
from anxietywatch_ml.pipelines.train import run_training, TrainingPipeline
from anxietywatch_ml.pipelines.predict import run_prediction, PredictionPipeline
from anxietywatch_ml.pipelines.model_pipeline import (
    ModelPipelineConfig,
    evaluate_pipeline,
    train_with_pipeline,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger(__name__)

console = Console()

app = typer.Typer(
    name="anxietywatch-ml",
    help="AnxietyWatch ML — Machine Learning pipeline (MVP bootstrap)",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file", exists=True
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """AnxietyWatch ML CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("anxietywatch_ml").setLevel(logging.DEBUG)

    # Load config early to validate
    if config:
        try:
            load_config(str(config))
            console.print(f"[green]Loaded config from {config}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load config: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file", exists=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save trained model"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed (overrides config)"
    ),
):
    """
    Train the baseline model on synthetic data.

    This uses SYNTHETIC DATA ONLY.
    The resulting model is an INFRASTRUCTURE BASELINE, not a clinical detector.
    """
    console.print("[bold]AnxietyWatch ML - Training Pipeline[/bold]")
    console.print("[yellow][!] DATA: SYNTHETIC - MODEL: INFRASTRUCTURE BASELINE[/yellow]")
    console.print("[yellow][!] NOT A CLINICAL ANXIETY DETECTOR[/yellow]")
    console.print()

    try:
        cfg = load_config(str(config) if config else None)
        if seed is not None:
            cfg["random_seed"] = seed

        result = run_training(str(config) if config else None, str(output) if output else None)

        console.print("\n[bold green]Training completed![/bold green]")
        console.print(f"  Train samples: {result.n_train}")
        console.print(f"  Val samples:   {result.n_val}")
        console.print(f"  Test samples:  {result.n_test}")
        console.print(f"  Features:      {len(result.feature_names)}")
        console.print(f"\n[bold]Test Metrics:[/bold]")
        for name, value in result.test_metrics.metrics.items():
            console.print(f"  {name}: {value:.4f}")

        if output:
            console.print(f"\n[green]Model saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        logger.exception("Training error")
        raise typer.Exit(1)


@app.command()
def predict(
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model", exists=True),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file", exists=True
    ),
    input_data: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Path to input data (CSV/Parquet)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save predictions"
    ),
):
    """
    Run prediction on data using a trained model.
    """
    console.print("[bold]AnxietyWatch ML - Prediction Pipeline[/bold]")
    console.print("[yellow][!] MODEL: INFRASTRUCTURE BASELINE - NOT CLINICAL[/yellow]")
    console.print()

    try:
        cfg = load_config(str(config) if config else None)
        pipeline = PredictionPipeline(cfg, model)

        if input_data:
            # Load input data
            if input_data.suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(input_data)
            elif input_data.suffix in [".parquet", ".pq"]:
                import pandas as pd
                df = pd.read_parquet(input_data)
            else:
                console.print(f"[red]Unsupported file format: {input_data.suffix}[/red]")
                raise typer.Exit(1)

            result = pipeline.run_from_dataframe(df)
        else:
            # Use synthetic data for demo
            from anxietywatch_ml.data.synthetic import create_generator
            generator = create_generator(cfg)
            batches, _ = generator.generate_dataset()
            result = pipeline.run(batches)

        console.print(f"\n[bold]Predictions:[/bold] {len(result.predictions)} windows")
        console.print(f"  Positive: {(result.predictions['prediction'] == 1).sum()}")
        console.print(f"  Negative: {(result.predictions['prediction'] == 0).sum()}")

        if output:
            result.predictions.to_csv(output, index=False)
            console.print(f"\n[green]Predictions saved to {output}[/green]")

        if result.metrics:
            console.print(f"\n[bold]Evaluation:[/bold]")
            for name, value in result.metrics.metrics.items():
                console.print(f"  {name}: {value:.4f}")

    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        logger.exception("Prediction error")
        raise typer.Exit(1)


@app.command()
def generate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file", exists=True
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, parquet, json"),
    batches: int = typer.Option(50, "--batches", "-b", help="Number of batches to generate"),
):
    """
    Generate synthetic telemetry data.

    Output is CLEARLY MARKED as synthetic.
    """
    console.print("[bold]AnxietyWatch ML - Synthetic Data Generator[/bold]")
    console.print("[yellow][!] DATA: SYNTHETIC - FOR PIPELINE TESTING ONLY[/yellow]")
    console.print()

    try:
        cfg = load_config(str(config) if config else None)
        generator = create_generator(cfg)

        # Override batch count if needed
        original_n_users = cfg.get("synthetic", {}).get("n_users", 10)
        original_n_sessions = cfg.get("synthetic", {}).get("n_sessions_per_user", 5)

        # Generate data
        df = generator.generate_dataframe()

        # Save
        output.parent.mkdir(parents=True, exist_ok=True)
        if format == "csv":
            df.to_csv(output, index=False)
        elif format == "parquet":
            df.to_parquet(output, index=False)
        elif format == "json":
            df.to_json(output, orient="records", date_format="iso")
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[green]Generated {len(df)} samples[/green]")
        console.print(f"  Users: {df['user_id'].nunique()}")
        console.print(f"  Sessions: {df['session_id'].nunique()}")
        console.print(f"  Batches: {df['batch_id'].nunique()}")
        console.print(f"  Saved to: {output}")

    except Exception as e:
        console.print(f"[red]Generation failed: {e}[/red]")
        logger.exception("Generation error")
        raise typer.Exit(1)


@app.command()
def validate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file", exists=True
    ),
    data: Path = typer.Option(..., "--data", "-d", help="Path to data file to validate", exists=True),
):
    """
    Validate telemetry data against the ML contract.
    """
    console.print("[bold]AnxietyWatch ML - Data Validation[/bold]")
    console.print()

    try:
        cfg = load_config(str(config) if config else None)

        # Load data
        if data.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(data)
        elif data.suffix in [".parquet", ".pq"]:
            import pandas as pd
            df = pd.read_parquet(data)
        else:
            console.print(f"[red]Unsupported file format: {data.suffix}[/red]")
            raise typer.Exit(1)

        from anxietywatch_ml.data.validation import validate_dataframe, log_validation_result
        result = validate_dataframe(df)
        log_validation_result(result, "Data validation")

        if result.is_valid:
            console.print("[green]Validation PASSED[/green]")
        else:
            console.print("[red]Validation FAILED[/red]")
            for err in result.errors:
                console.print(f"  [red]ERROR:[/red] {err}")

        for warn in result.warnings:
            console.print(f"  [yellow]WARNING:[/yellow] {warn}")

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        logger.exception("Validation error")
        raise typer.Exit(1)


@app.command()
def smoke(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config YAML file", exists=True
    ),
):
    """
    Run end-to-end smoke test: generate -> preprocess -> features -> train -> predict.
    """
    console.print("[bold]AnxietyWatch ML - Smoke Test[/bold]")
    console.print("[yellow][!] FULL PIPELINE TEST WITH SYNTHETIC DATA[/yellow]")
    console.print()

    try:
        cfg = load_config(str(config) if config else None)

        # Generate
        console.print("1/5 Generating synthetic data...")
        generator = create_generator(cfg)
        batches, anomaly_sessions = generator.generate_dataset()
        console.print(f"   Generated {len(batches)} batches, {sum(anomaly_sessions.values())} anomaly sessions")

        # Preprocess
        console.print("2/5 Preprocessing...")
        from anxietywatch_ml.preprocessing.pipeline import create_pipeline
        preprocessing = create_pipeline(cfg)
        windowed = preprocessing.run(batches)
        console.print(f"   Created {len(windowed.windows)} windows")

        # Features
        console.print("3/5 Building features...")
        from anxietywatch_ml.features.builder import create_feature_builder
        feature_builder = create_feature_builder(cfg)
        X = feature_builder.build(windowed.windows)
        console.print(f"   Feature matrix: {X.shape}")

        # Labels
        from anxietywatch_ml.pipelines.train import TrainingPipeline
        pipeline = TrainingPipeline(cfg)
        y = pipeline._create_labels(windowed, anomaly_sessions)
        console.print(f"   Labels: {y.sum()} positive, {len(y) - y.sum()} negative")

        # Group-aware split + training
        console.print("4/5 Group-aware split + training...")
        from anxietywatch_ml.pipelines.model_pipeline import (
            ModelPipelineConfig,
            evaluate_pipeline,
            train_with_pipeline,
        )

        training_cfg = cfg.get("training", {})

        group_by = training_cfg.get(
            "group_by",
            "session",
        )

        group_key = (
            "session_id"
            if group_by == "session"
            else "user_id"
        )

        groups = pd.Series(
            [
                meta[group_key]
                for meta in windowed.window_metadata
            ],
            index=X.index,
        )

        pipeline_config = ModelPipelineConfig(
            # Smoke test deliberately exercises a real finite-matrix estimator.
            model_type="logistic_regression",
            group_by=group_by,
            test_size=training_cfg.get("test_size", 0.2),
            val_size=training_cfg.get("val_size", 0.1),
            random_state=training_cfg.get(
                "random_state",
                cfg.get("random_seed", 42),
            ),
        )

        bundle = train_with_pipeline(
            X=X,
            y=y,
            group_column=groups,
            config=pipeline_config,
        )

        console.print(
            "   Split: "
            f"train={len(bundle.split_result.train_indices)}, "
            f"val={len(bundle.split_result.val_indices)}, "
            f"test={len(bundle.split_result.test_indices)}"
        )

        console.print(
            "   Groups: "
            f"train={len(bundle.split_result.train_groups)}, "
            f"val={len(bundle.split_result.val_groups)}, "
            f"test={len(bundle.split_result.test_groups)}"
        )

        console.print("5/5 Evaluating plumbing...")

        results = evaluate_pipeline(
            bundle,
            X,
            y,
        )

        for split_name in ("train", "val", "test"):
            result = results[split_name]

            if result is None:
                continue

            metrics = result["metrics"]

            accuracy = metrics.get("accuracy")
            roc_auc = metrics.get("roc_auc")

            console.print(
                f"   {split_name}: "
                f"accuracy={accuracy:.3f} "
                f"roc_auc="
                f"{roc_auc:.3f}"
                if roc_auc is not None
                else f"   {split_name}: roc_auc=N/A"
            )

        console.print("\n[bold green][OK] Smoke test PASSED[/bold green]")
        console.print("[yellow]NOTE: This smoke test uses train/val/test split on SYNTHETIC data.[/yellow]")
        console.print("[yellow]      Metrics shown are for plumbing verification only, NOT clinical validation.[/yellow]")
        console.print("Pipeline is functional!")

    except Exception as e:
        console.print(f"\n[bold red][FAIL] Smoke test FAILED: {e}[/bold red]")
        logger.exception("Smoke test error")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()