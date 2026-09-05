"""Command-line entry point for the reproducible Pipeline stages."""

import argparse
import logging
import sys

from src.pipeline.build_pipeline import (
    CORE_FOLDS,
    MODELS,
    VARIANTS,
    run_all_stages,
    run_baseline_hpo,
    run_core_stage,
    run_hpo_stage,
    run_shap_stage,
    run_tuned_hpo,
)
from src.pipeline.qa_checks import write_run_matrix
from src.pipeline.run_shap import save_sample_keys
from src.pipeline.summarize import summarize_core
from src.utilities import load_data


def _parser() -> argparse.ArgumentParser:
    """Build the explicit Pipeline subcommand parser."""
    parser = argparse.ArgumentParser(description="Run the locked SHAP stability pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample-shap", help="Generate semantic SHAP sample keys")
    sample.add_argument("fold", choices=CORE_FOLDS)
    baseline = subparsers.add_parser("baseline-hpo", help="Evaluate one untuned HPO baseline")
    baseline.add_argument("model", choices=MODELS)
    hpo = subparsers.add_parser("hpo", help="Run configured Optuna trials")
    hpo.add_argument("model", choices=MODELS)
    tuned = subparsers.add_parser("tuned-hpo", help="Evaluate one frozen HPO configuration")
    tuned.add_argument("model", choices=MODELS)
    core = subparsers.add_parser("train-core", help="Train one tuned core model")
    core.add_argument("model", choices=MODELS)
    core.add_argument("variant", choices=VARIANTS)
    core.add_argument("fold", choices=CORE_FOLDS)
    shap = subparsers.add_parser("shap", help="Generate SHAP artifacts for one core model")
    shap.add_argument("model", choices=MODELS)
    shap.add_argument("variant", choices=VARIANTS)
    shap.add_argument("fold", choices=CORE_FOLDS)
    subparsers.add_parser("check-matrix", help="Validate and write the experiment matrix")
    subparsers.add_parser("summarize", help="Validate and write canonical summary tables")
    subparsers.add_parser("all", help="Run all stages, validate, and summarize")
    return parser


def main(argv: list[str]) -> int:
    """Dispatch one explicit Pipeline subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _parser().parse_args(argv)
    command = arguments.command
    if command == "sample-shap":
        save_sample_keys(arguments.fold, load_data(arguments.fold, "A"))
    elif command == "baseline-hpo":
        run_baseline_hpo(arguments.model)
    elif command == "hpo":
        run_hpo_stage(arguments.model)
    elif command == "tuned-hpo":
        run_tuned_hpo(arguments.model)
    elif command == "train-core":
        run_core_stage(arguments.model, arguments.variant, arguments.fold)
    elif command == "shap":
        run_shap_stage(arguments.model, arguments.variant, arguments.fold)
    elif command == "check-matrix":
        write_run_matrix()
    elif command == "summarize":
        summarize_core()
    elif command == "all":
        run_all_stages()
        write_run_matrix()
        summarize_core()
    else:
        raise ValueError(f"Unsupported command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
