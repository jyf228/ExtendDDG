import pandas as pd
from openai import OpenAI

from extendddg import ExtendDDG, GPTEvaluator
from extendddg.utils import get_sample

api_key = "your-api-key"
client = OpenAI(api_key=api_key)
model_name = "gpt-4o-mini"

extend_ddg = ExtendDDG(client=client, model_name=model_name)

# ---------------------------------------------------------
# 1. Load dataset and documentation (if available)
# ---------------------------------------------------------
dataset_file = "datasets/rls_dataset_example.csv"
codebook_file = "docs/rls_codebook.csv"  # Set to None if not available
documentation_file ="docs/rls_methodology_report.pdf" # Set to None if not available

# Set dataset title and original dataset description
title = "2023-24 Religious Landscape Study (RLS) Dataset"
original_description = (
    "This Pew Research Center survey was conducted among a nationally representative "
    "sample of adults to estimate the U.S. population’s religious composition, beliefs, "
    "and practices."
)

# Read dataset
csv_df = pd.read_csv(dataset_file)

# ---------------------------------------------------------
# 2. Column weighting (optional) and sampling
# ---------------------------------------------------------
# This is specific to the RLS dataset, but can be adapted for other datasets.
# Here we downweight the replicate weights (REPWT_) columns.
col_weights = [
    0.01 if col.startswith("REPWT_") else 1.0
    for col in csv_df.columns
]

# Sample columns
col_sample_df, col_dataset_sample = get_sample(
    csv_df,
    row_sample_size=10000,
    col_sample_size=120,
    col_weights=col_weights,
)

# ---------------------------------------------------------
# 3. Data-Driven Profiling
# ---------------------------------------------------------
basic_profile, structural_profile = extend_ddg.profile_dataframe(col_sample_df)
print("**** Basic Profile ****\n", basic_profile)
print("\n**** Structural Profile ****\n", structural_profile)

# ---------------------------------------------------------
# 4. Codebook Profiling (optional)
# ---------------------------------------------------------
# Sample rows for codebook and semantic profiling
sample_df, dataset_sample = get_sample(
    col_sample_df,
    row_sample_size=100,
)

# Generate codebook profile if codebook is provided
codebook_profile = {}
if codebook_file:
    codebook_profile = extend_ddg.profile_codebook(
        dataset_df=sample_df,
        codebook_file=codebook_file,
    )
    print("\n**** Codebook Profile ****\n", codebook_profile)

# ---------------------------------------------------------
# 5. Semantic Profiling
# ----------------------------------------------------------
semantic_profile_details = extend_ddg.analyze_semantics(
    sample_df,
    codebook_profile=codebook_profile if codebook_profile else None,
)
semantic_profile = "\n".join(
    section for section in [structural_profile, semantic_profile_details] if section
)
print("\n**** Semantic Profile ****\n", semantic_profile)

# ---------------------------------------------------------
# 6. Topic generation
# ---------------------------------------------------------
data_topic = extend_ddg.generate_topic(
    title=title,
    original_description=original_description,
    dataset_sample=dataset_sample,
)
print("\n**** Data Topic ****\n", data_topic)

# ---------------------------------------------------------
# 7. Documentation Profile (optional)
# ---------------------------------------------------------
documentation_profile = {}
if documentation_file:
    documentation_profile = extend_ddg.profile_documentation(documentation_file)
    breakpoint()

print("\n**** Documentation Profile ****\n", documentation_profile)

# ---------------------------------------------------------
# 8. User-focused description
# ---------------------------------------------------------
prompt, description = extend_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
    documentation_profile=documentation_profile,
    use_documentation_profile=True if documentation_file else False,
    codebook_profile=codebook_profile,
    use_codebook_profile=True if codebook_file else False,
)
breakpoint()

# ---------------------------------------------------------
# 9. Search-focused description
# ---------------------------------------------------------
search_prompt, search_focused_description = extend_ddg.expand_description_for_search(
    description=description,
    topic=data_topic,
)

# ---------------------------------------------------------
# 10. Evaluation
# ---------------------------------------------------------
extend_ddg.set_evaluator(GPTEvaluator(gpt4_api_key=api_key))

general_score = extend_ddg.evaluate_description(description)
search_score = extend_ddg.evaluate_description(search_focused_description)

print("\n**** Score of the General Description ****\n", general_score)
print("\n**** Score of the Search-Focused Description ****\n", search_score)

# evaluation_metrics = extend_ddg.evaluation_metrics(description, original_description)
