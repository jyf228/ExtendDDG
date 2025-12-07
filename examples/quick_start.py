import os
import pandas as pd
from openai import OpenAI

from extendddg import ExtendDDG, GPTEvaluator
from extendddg.parsing.codebook import CodebookParser
from extendddg.parsing.documentation import DocumentationParser
from extendddg.utils import get_sample


# ---------------------------------------------------------
# 1. Setup: Load API key safely
# ---------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your_key'")

client = OpenAI(api_key=api_key)
model_name = "gpt-4o-mini"

# Initialize ExtendDDG
extend_ddg = ExtendDDG(client=client, model_name=model_name)


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
# NOTE: This is a subset of the full dataset (150 rows) due to GitHub file-size limits.
# You will probably want to test with the full dataset.
csv_file = "datasets/rls_dataset_example.csv"
title = "2023-24 Religious Landscape Study (RLS) Dataset"

original_description = (
    "This Pew Research Center survey was conducted among a nationally representative sample "
    "of adults to estimate the U.S. population’s religious composition, beliefs and practices."
)

csv_df = pd.read_csv(csv_file)


# ---------------------------------------------------------
# 3. Column weights for sampling
# ---------------------------------------------------------
col_weights = []
for col in csv_df.columns:
    if col.startswith("REPWT_"):
        col_weights.append(0.01)  # Lower weight for replicate columns
    else:
        col_weights.append(1.0)   # Normal weight


# ---------------------------------------------------------
# 4. Column sampling (full rows, sampled columns)
# ---------------------------------------------------------
col_sample_df, col_dataset_sample = get_sample(
    csv_df,
    row_sample_size=36910,
    col_sample_size=160,
    col_weights=col_weights,
)


# ---------------------------------------------------------
# 5. Basic + Structural Profiles
# ---------------------------------------------------------
basic_profile, structural_profile = extend_ddg.profile_dataframe(col_sample_df)
print("**** Basic Profile ****\n", basic_profile)
print("\n**** Structural Profile ****\n", structural_profile)


# ---------------------------------------------------------
# 6. Sample rows for semantic profiling
# ---------------------------------------------------------
sample_df, dataset_sample = get_sample(
    col_sample_df,
    row_sample_size=100,
)


# ---------------------------------------------------------
# 7. Codebook Parser
# ---------------------------------------------------------
# TODO: Move this into ExtendDDG as an integrated step (future)
codebook_parser = CodebookParser()
codebook_df = codebook_parser.extract_variables("datasets/rls_codebook.csv")


# ---------------------------------------------------------
# 8. Semantic Profile (now uses codebook)
# ---------------------------------------------------------
semantic_profile_details = extend_ddg.analyze_semantics(
    sample_df,
    codebook=codebook_df  # ← Our implemented update
)

semantic_profile = "\n".join(
    section for section in [structural_profile, semantic_profile_details] if section
)

print("\n**** Semantic Profile ****\n", semantic_profile)


# ---------------------------------------------------------
# 9. Topic Generation
# ---------------------------------------------------------
data_topic = extend_ddg.generate_topic(
    title=title,
    original_description=original_description,
    dataset_sample=dataset_sample,
)

print("\n**** Data Topic ****\n", data_topic)


# ---------------------------------------------------------
# 10. Documentation Parser
# ---------------------------------------------------------
# TODO: Improve methodology extraction, add OCR fallback
doc_parser = DocumentationParser()
documentation_profile = doc_parser.parse("datasets/rls_methodology_report.pdf")

print("\n**** Documentation Profile ****\n", documentation_profile)


# ---------------------------------------------------------
# 11. Final dataset description
# ---------------------------------------------------------
prompt, description = extend_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
    documentation_profile=documentation_profile,     # ← Implemented feature
    use_documentation_profile=True,                  # ← Implemented feature
)

print("\n**** Description Prompt ****\n", prompt)
print("\n**** Dataset Description ****\n", description)


# ---------------------------------------------------------
# 12. Search-focused description
# ---------------------------------------------------------
search_prompt, search_focused_description = extend_ddg.expand_description_for_search(
    description=description,
    topic=data_topic,
)

print("\n**** Search Prompt ****\n", search_prompt)
print("\n**** Search-Focused Description ****\n", search_focused_description)


# ---------------------------------------------------------
# 13. Evaluator + Scoring
# ---------------------------------------------------------
extend_ddg.set_evaluator(GPTEvaluator(gpt4_api_key=api_key))

# Score the descriptions
general_score = extend_ddg.evaluate_description(description)
search_score = extend_ddg.evaluate_description(search_focused_description)

print("\n**** Score: General Description ****\n", general_score)
print("\n**** Score: Search Description ****\n", search_score)
