import os
import pandas as pd
from openai import OpenAI

from extendddg import ExtendDDG, GPTEvaluator
from extendddg.parsing.codebook import CodebookParser
from extendddg.parsing.documentation import DocumentationParser
from extendddg.utils import get_sample

# ---------------------------------------------------------
# 1. Setup: Load API key
# ---------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your_key'")

client = OpenAI(api_key=api_key)
model_name = "gpt-4o-mini"

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "datasets")

extend_ddg = ExtendDDG(client=client, model_name=model_name)

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
csv_file = os.path.join(DATA_DIR, "rls_dataset_example.csv")
title = "2023-24 Religious Landscape Study (RLS) Dataset"

original_description = (
    "This Pew Research Center survey was conducted among a nationally representative "
    "sample of adults to estimate the U.S. population’s religious composition, beliefs, "
    "and practices."
)

csv_df = pd.read_csv(csv_file)

# ---------------------------------------------------------
# 3. Column weights
# ---------------------------------------------------------
col_weights = [
    0.01 if col.startswith("REPWT_") else 1.0
    for col in csv_df.columns
]

# ---------------------------------------------------------
# 4. Column sampling
# ---------------------------------------------------------
col_sample_df, col_dataset_sample = get_sample(
    csv_df,
    row_sample_size=10000,
    col_sample_size=120,
    col_weights=col_weights,
)

# ---------------------------------------------------------
# 5. Basic + Structural Profiles
# ---------------------------------------------------------
basic_profile, structural_profile = extend_ddg.profile_dataframe(col_sample_df)

# ---------------------------------------------------------
# 6. Semantic profiling
# ---------------------------------------------------------
sample_df, dataset_sample = get_sample(
    col_sample_df,
    row_sample_size=100,
)

# Codebook
codebook_parser = CodebookParser()
codebook_df = codebook_parser.extract_variables(
    os.path.join(DATA_DIR, "rls_codebook.csv")
)

semantic_profile_details = extend_ddg.analyze_semantics(
    sample_df,
    codebook=codebook_df,
)

semantic_profile = "\n".join(
    section for section in [structural_profile, semantic_profile_details] if section
)

# ---------------------------------------------------------
# 7. Topic generation
# ---------------------------------------------------------
data_topic = extend_ddg.generate_topic(
    title=title,
    original_description=original_description,
    dataset_sample=dataset_sample,
)

# ---------------------------------------------------------
# 8. Documentation Profile (optional)
# ---------------------------------------------------------
documentation_file = os.path.join(DATA_DIR, "rls_methodology_report.pdf")
documentation_path = documentation_file if os.path.exists(documentation_file) else None

# ---------------------------------------------------------
# 9. Final dataset description
# ---------------------------------------------------------
prompt, description = extend_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
    documentation_path=documentation_path,
)

# ---------------------------------------------------------
# 10. Search-focused description
# ---------------------------------------------------------
search_prompt, search_focused_description = extend_ddg.expand_description_for_search(
    description=description,
    topic=data_topic,
)

# ---------------------------------------------------------
# 11. Evaluation
# ---------------------------------------------------------
extend_ddg.set_evaluator(GPTEvaluator(gpt4_api_key=api_key))

general_score = extend_ddg.evaluate_description(description)
search_score = extend_ddg.evaluate_description(search_focused_description)
