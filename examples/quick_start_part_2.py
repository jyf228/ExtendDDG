"""Variable-level alignment check using the RLS codebook output."""

import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

from extendddg.evaluation import evaluate_variable_alignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DESC_PATH = ROOT / "examples/test_outputs/rls_output_codebook.txt"
CODEBOOK_PATH = ROOT / "examples/codebooks/rls_codebook.csv"
MODEL_NAME = "gpt-4o-mini"


def load_description(path: Path) -> str:
    """Pull the generated description out of the saved test output."""
    raw_text = path.read_text(encoding="utf-8")
    parts = raw_text.split("**** Dataset Description ****")
    return parts[1].strip() if len(parts) > 1 else raw_text.strip()


def main() -> None:
    description = load_description(DESC_PATH)
    codebook_df = pd.read_csv(CODEBOOK_PATH)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment before running.")

    client = OpenAI(api_key=api_key)

    result = evaluate_variable_alignment(
        description=description,
        codebook=codebook_df,
        client=client,
        sample_size=5,
        model_name=MODEL_NAME,
    )

    print("Support rate:", result["support_rate"])
    for entry in result["results"]:
        print(entry)


if __name__ == "__main__":
    main()
