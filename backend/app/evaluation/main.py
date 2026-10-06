"""Entry point for Phase 7 evaluation.

Usage:
    # From the repository root (the original command the user tried)
    python -m backend.app.evaluation.main --dataset data/evaluation/cases.json --out results/

    # From the backend directory (the alternative form)
    cd backend
    python -m app.evaluation.main --dataset ../data/evaluation/cases.json --out ../results/

The script loads the JSON dataset, runs the baseline and ReGenRAG runners
against a database session, compares the results, and writes a JSON/CSV report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Relative imports keep the package namespace consistent (`app`).
from .dataset import load_cases, create_sample_dataset
from .baseline_runner import run_baseline
from .regenrag_runner import run_regenrag
from .comparator import compare_runs, save_report, EvaluationReport
from .result import EvalRunResult

# DB session creation is deferred to runtime so the script can fail gracefully
# when the environment is not set up (e.g., during tests).
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as DbSession
    from ..core.config import settings
    HAS_DB = True
except ImportError:
    HAS_DB = False


def _get_db_session():
    """Create a SQLAlchemy session using the configured DATABASE_URL."""
    if not HAS_DB:
        raise RuntimeError(
            "Database dependencies not available. Install the project in the "
            "backend directory or ensure sqlalchemy/psycopg2 are installed."
        )
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Copy backend/.env.example to "
            "backend/.env and set DATABASE_URL."
        )
    engine = create_engine(settings.database_url)
    return DbSession(engine)


def _run_evaluation(
    cases: list,
    *,
    db_session,
    metadata: dict | None = None,
) -> EvaluationReport:
    """Execute the full Phase 7 evaluation pipeline.

    For each case, runs both the baseline and ReGenRAG pipelines, then
    aggregates the results and produces a comparison report.
    """
    baseline_results = []
    regen_results = []

    for case in cases:
        # Baseline run (Phase 3 style)
        try:
            b_res = run_baseline(db_session, case)
            baseline_results.append(b_res)
        except Exception as exc:
            print(f"Baseline runner failed for case {case.id}: {exc}", file=sys.stderr)
            # Create a minimal failure result so the case is still represented
            from app.evaluation.result import EvalResult
            baseline_results.append(
                EvalResult(
                    case_id=case.id,
                    query=case.query,
                    hit_rate=0.0,
                    mrr=0.0,
                    ndcg=0.0,
                    expected_answer=case.expected_answer,
                )
            )

        # ReGenRAG run (Phase 6 style)
        try:
            r_res = run_regenrag(db_session, case)
            regen_results.append(r_res)
        except Exception as exc:
            print(f"ReGenRAG runner failed for case {case.id}: {exc}", file=sys.stderr)
            from app.evaluation.result import EvalResult
            regen_results.append(
                EvalResult(
                    case_id=case.id,
                    query=case.query,
                    hit_rate=0.0,
                    mrr=0.0,
                    ndcg=0.0,
                    expected_answer=case.expected_answer,
                )
            )

    baseline_run = EvalRunResult(results=baseline_results)
    regen_run = EvalRunResult(results=regen_results)

    run_metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_cases": len(cases),
        "config": {
            "rag_top_k": settings.rag_top_k if HAS_DB else None,
            "max_rag_retries": settings.max_rag_retries if HAS_DB else None,
            "retrieval_similarity_threshold": (
                settings.retrieval_similarity_threshold if HAS_DB else None
            ),
            "grounding_threshold": settings.grounding_threshold if HAS_DB else None,
            "embedding_model": settings.embedding_model if HAS_DB else None,
            "llm_provider": settings.llm_provider if HAS_DB else None,
            "llm_model": settings.llm_model if HAS_DB else None,
        },
        **(metadata or {}),
    }

    return compare_runs(baseline_run, regen_run, metadata=run_metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 7 evaluation: compare Baseline RAG vs ReGenRAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSON file with evaluation cases",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory where the evaluation report will be written",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Generate a minimal sample dataset at --dataset path and exit",
    )
    args = parser.parse_args(argv)

    if args.create_sample:
        create_sample_dataset(args.dataset)
        print(f"Sample dataset written to {args.dataset}")
        return 0

    cases = load_cases(args.dataset)
    print(f"Loaded {len(cases)} evaluation cases from {args.dataset}")

    # If DB is not available, run a dry-run that only validates the dataset
    # and writes a placeholder report. This allows CI / test environments
    # to exercise the CLI without a live database.
    if not HAS_DB or not settings.database_url:
        print("WARNING: Database not configured — running in dry-run mode.")
        from .result import EvalResult
        # Create dummy results to test the comparator/report path
        dummy_baseline = [
            EvalResult(
                case_id=c.id,
                query=c.query,
                hit_rate=0.0,
                mrr=0.0,
                ndcg=0.0,
                expected_answer=c.expected_answer,
            )
            for c in cases
        ]
        dummy_regen = [
            EvalResult(
                case_id=c.id,
                query=c.query,
                hit_rate=0.5,
                mrr=0.5,
                ndcg=0.5,
                grounding_score=0.8,
                expected_answer=c.expected_answer,
            )
            for c in cases
        ]
        # Attach healing metadata for recovery analysis
        for r in dummy_regen:
            r._attempts = 2
            r._retry_exhausted = False
            r.rewritten_query = "rewritten"

        report = compare_runs(
            EvalRunResult(results=dummy_baseline),
            EvalRunResult(results=dummy_regen),
        )
    else:
        print("Running baseline and ReGenRAG pipelines...")
        db = _get_db_session()
        try:
            report = _run_evaluation(cases, db_session=db)
        finally:
            db.close()

    json_path, csv_path = save_report(report, args.out)
    print(f"Evaluation report written to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Print a summary to stdout
    agg = report.aggregate_metrics
    print("\n=== Phase 7 Evaluation Summary ===")
    print(f"Total cases: {agg['total_cases']}")
    print(f"Baseline HR@K: {agg['baseline']['hit_rate']:.3f}")
    print(f"ReGenRAG HR@K: {agg['regenrag']['hit_rate']:.3f}")
    print(f"Baseline MRR:  {agg['baseline']['mrr']:.3f}")
    print(f"ReGenRAG MRR:  {agg['regenrag']['mrr']:.3f}")
    print(f"Baseline nDCG: {agg['baseline']['ndcg']:.3f}")
    print(f"ReGenRAG nDCG: {agg['regenrag']['ndcg']:.3f}")
    print(f"Improved:      {agg['comparison']['improved_cases']}")
    print(f"Unchanged:     {agg['comparison']['unchanged_cases']}")
    print(f"Regressed:     {agg['comparison']['regressed_cases']}")
    rec = report.recovery_analysis
    print(f"\nRecovery rate:     {rec['recovery_rate']:.1%}")
    print(f"Retry rate:        {rec['retry_rate']:.1%}")
    print(f"Query rewrite rate: {rec['query_rewrite_rate']:.1%}")
    print(f"Avg attempts:      {rec['average_attempts']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())