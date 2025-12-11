"""Run AutoDDG vs. ExtendDDG end-to-end across example datasets and save metrics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from autoddg import AutoDDG, GPTEvaluator
from autoddg.autoddg import SemanticProfiler as AutoSemanticProfiler
from openai import OpenAI

from extendddg import ExtendDDG
from extendddg.evaluation.metrics import (
    detailed_evaluate_description,
    redundancy_verbosity_metrics,
    unique_fact_ratio,
)
from extendddg.utils import get_sample

MODEL_NAME = "gpt-4o-mini"
ROW_SAMPLE = 5_000
COL_SAMPLE = 25  # limit for speed/cost during semantic profiling
SEMANTIC_ROW_SAMPLE = 40

DATASETS = [
    {
        "name": "rls",
        "csv": Path("examples/datasets/rls_dataset_example.csv"),
        "codebook": Path("examples/codebooks/rls_codebook.csv"),
        "documentation": Path("examples/docs/rls_methodology_report.pdf"),
        "title": "2023-24 Religious Landscape Study (RLS) Dataset",
        "original_description": (
            "This Pew Research Center survey was conducted among a nationally "
            "representative sample of adults to estimate the U.S. population's "
            "religious composition, beliefs, and practices."
        ),
    },
    {
        "name": "clark",
        "csv": Path("examples/datasets/clark_dataset.csv"),
        "codebook": None,
        "documentation": None,
        "title": "Clark Renal Cell Carcinoma Dataset",
        "original_description": (
            "Clinical dataset of renal cell carcinoma cases including demographics, "
            "tumor characteristics, staging, and outcomes."
        ),
    },
]


def extract_metrics(description: str, reference: str | None) -> Dict[str, Any]:
    detailed = detailed_evaluate_description(description, reference) if reference else {}
    redundancy = redundancy_verbosity_metrics(description)
    redundancy["unique_fact_ratio"] = unique_fact_ratio(description)
    redundancy["word_count"] = len(description.split())
    return {"detailed": detailed, "redundancy": redundancy}


def run_models_for_dataset(
    name: str,
    df: pd.DataFrame,
    *,
    codebook_path: Path | None,
    documentation_path: Path | None,
    title: str,
    original_description: str | None,
    client: OpenAI,
) -> Dict[str, Any]:
    col_weights = (
        [0.01 if str(col).startswith("REPWT_") else 1.0 for col in df.columns]
        if name == "rls"
        else None
    )

    sample_df, dataset_sample = get_sample(
        df,
        row_sample_size=min(ROW_SAMPLE, len(df)),
        col_sample_size=min(COL_SAMPLE, df.shape[1]),
        col_weights=col_weights,
    )

    semantic_df, dataset_sample_for_desc = get_sample(
        sample_df, row_sample_size=min(SEMANTIC_ROW_SAMPLE, len(sample_df))
    )

    # ExtendDDG pipeline
    extend = ExtendDDG(client=client, model_name=MODEL_NAME)
    basic_profile, structural_profile = extend.profile_dataframe(sample_df)
    codebook_profile: Dict[str, Dict[str, Any]] = {}
    if codebook_path:
        codebook_profile = extend.profile_codebook(dataset_df=semantic_df, codebook_file=str(codebook_path))
    semantic_profile = extend.analyze_semantics(semantic_df, codebook_profile=codebook_profile or None)
    topic = extend.generate_topic(title=title, original_description=original_description or "", dataset_sample=dataset_sample_for_desc)
    documentation_profile: Dict[str, Any] = {}
    if documentation_path:
        documentation_profile = extend.profile_documentation(str(documentation_path))
    _, extend_description = extend.describe_dataset(
        dataset_sample=dataset_sample_for_desc,
        dataset_profile=basic_profile,
        use_profile=True,
        semantic_profile=semantic_profile,
        use_semantic_profile=True,
        data_topic=topic,
        use_topic=True,
        documentation_profile=documentation_profile,
        use_documentation_profile=bool(documentation_profile),
        codebook_profile=codebook_profile,
        use_codebook_profile=bool(codebook_profile),
    )
    _, extend_search_description = extend.expand_description_for_search(extend_description, topic)
    extend.set_evaluator(GPTEvaluator(gpt4_api_key=os.environ["OPENAI_API_KEY"]))
    extend_general_score = extend.evaluate_description(extend_description)
    extend_search_score = extend.evaluate_description(extend_search_description)

    # AutoDDG pipeline
    auto = AutoDDG(client=client, model_name=MODEL_NAME, description_words=200)
    auto_semantic = AutoSemanticProfiler(client=client, model_name=MODEL_NAME)
    auto_semantic_profile = auto_semantic.analyze_dataframe(semantic_df)
    _, auto_description = auto.describe_dataset(
        dataset_sample=dataset_sample_for_desc,
        dataset_profile=basic_profile,
        use_profile=True,
        semantic_profile=auto_semantic_profile,
        use_semantic_profile=True,
        data_topic=topic,
        use_topic=True,
    )
    _, auto_search_description = auto.expand_description_for_search(auto_description, topic)
    auto.set_evaluator(GPTEvaluator(gpt4_api_key=os.environ["OPENAI_API_KEY"]))
    auto_general_score = auto.evaluate_description(auto_description)
    auto_search_score = auto.evaluate_description(auto_search_description)

    return {
        "dataset": name,
        "title": title,
        "topic": topic,
        "profiles": {
            "basic": basic_profile,
            "structural": structural_profile,
            "semantic_extend": semantic_profile,
            "semantic_auto": auto_semantic_profile,
        },
        "models": {
            "ExtendDDG": {
                "description": extend_description,
                "search_description": extend_search_description,
                "general_score": extend_general_score,
                "search_score": extend_search_score,
                "metrics": extract_metrics(extend_description, original_description),
            },
            "AutoDDG": {
                "description": auto_description,
                "search_description": auto_search_description,
                "general_score": auto_general_score,
                "search_score": auto_search_score,
                "metrics": extract_metrics(auto_description, original_description),
            },
        },
    }


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running.")

    client = OpenAI(api_key=api_key)
    output_dir = Path("examples/run_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for ds in DATASETS:
        print(f"Running dataset: {ds['name']}")
        df = pd.read_csv(ds["csv"])
        result = run_models_for_dataset(
            ds["name"],
            df,
            codebook_path=ds.get("codebook"),
            documentation_path=ds.get("documentation"),
            title=ds["title"],
            original_description=ds.get("original_description"),
            client=client,
        )
        output_path = output_dir / f"{ds['name']}_results.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results.append(result)
        print(f"Saved {output_path}")

    combined_path = output_dir / "combined_results.json"
    combined_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved combined results to {combined_path}")


if __name__ == "__main__":
    main()
