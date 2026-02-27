"""
run_experiments.py
==================
Entry point for running one or more experiments.

Usage
-----
Single experiment::

    python run_experiments.py experiments/diffwave_400ep.yaml

Multiple experiments (batch file)::

    python run_experiments.py --batch batch_run.yaml

Multiple experiments (multiple files)::

    python run_experiments.py experiments/diffwave.yaml experiments/timegan.yaml

Programmatic::

    from runner import run_experiment
    run_experiment("experiments/diffwave_400ep.yaml")
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

from runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ML experiment(s) from YAML config files."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Path(s) to experiment YAML config file(s).",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to a batch YAML listing multiple experiment configs.",
    )
    args = parser.parse_args()

    config_paths: list[Path] = []

    # Collect configs from --batch
    if args.batch:
        batch_path = Path(args.batch)
        with open(batch_path, encoding="utf-8") as f:
            batch = yaml.safe_load(f)
        batch_dir = batch_path.parent
        for rel in batch.get("experiments", []):
            config_paths.append((batch_dir / rel).resolve())

    # Collect configs from positional args
    for c in args.configs:
        config_paths.append(Path(c).resolve())

    if not config_paths:
        parser.print_help()
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Experiment Suite – {len(config_paths)} experiment(s) queued")
    print(f"{'='*60}")

    results = {}
    for i, cfg_path in enumerate(config_paths, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(config_paths)}] {cfg_path.name}")
        print(f"{'─'*60}")
        try:
            output_dir = run_experiment(cfg_path)
            results[cfg_path.name] = ("SUCCESS", output_dir)
        except Exception as e:
            traceback.print_exc()
            results[cfg_path.name] = ("FAILED", str(e))

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, (status, info) in results.items():
        icon = "✓" if status == "SUCCESS" else "✗"
        print(f"  {icon} {name}: {status} → {info}")


if __name__ == "__main__":
    main()