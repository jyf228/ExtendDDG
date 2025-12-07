import pandas as pd
from openai import OpenAI

from extendddg import ExtendDDG, GPTEvaluator
from extendddg.utils import get_sample

# Setup OpenAI client
api_key = "you-api-key"
client = OpenAI(api_key=api_key)
model_name = "gpt-4o-mini"

# Initialize ExtendDDG
extend_ddg = ExtendDDG(client=client, model_name=model_name)

# Load dataset
csv_file = "datasets/rls_dataset_example.csv"
title = "2023-24 Religious Landscape Study (RLS) Dataset"
original_description = (
    "This Pew Research Center survey was conducted among a nationally representative sample of adults "
    "to provide estimates of the U.S. population’s religious composition, beliefs and practices."
)
csv_df = pd.read_csv(csv_file)

# Create column weights
col_weights = []
for col in csv_df.columns:
    if col.startswith('REPWT_'):
        col_weights.append(0.01)  # Lower weight for replicate columns
    else:
        col_weights.append(1.0)  # Normal weight for other columns

# Sample columns but use all rows for data profiling
col_sample_df, col_dataset_sample = get_sample(
    csv_df,
    row_sample_size=36910,
    col_sample_size=160,
    col_weights=col_weights,
)

# Generate data profiles
basic_profile, structural_profile = extend_ddg.profile_dataframe(col_sample_df)
print("**** Basic Profile ****\n", basic_profile)
print("\n**** Structural Profile ****\n", structural_profile)

# Sample rows for semantic profiling
sample_df, dataset_sample = get_sample(
    col_sample_df,
    row_sample_size=100,
)

# Generate semantic profile
semantic_profile_details = extend_ddg.analyze_semantics(
    sample_df,
    codebook_path="datasets/rls_codebook.csv"
)
semantic_profile = "\n".join(
    section for section in [structural_profile, semantic_profile_details] if section
)
print("\n**** Semantic Profile ****\n", semantic_profile)

# Generate topic
data_topic = extend_ddg.generate_topic(
    title=title,
    original_description=original_description,
    dataset_sample=dataset_sample,
)

print("\n**** Data Topic ****\n", data_topic)

# TODO: Call DocParser if additional documentation is available

# TODO: Generate documentation profile
# print("\n**** Documentation Profile ****\n", documentation_profile)

# General description
prompt, description = extend_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
    # documentation_profile=documentation_profile,  # TODO
    # use_documentation_profile=True,  # TODO
)

print("\n**** Description Prompt ****\n", prompt)
print("\n**** Dataset Description ****\n", description)

# Search-focused description
search_prompt, search_focused_description = extend_ddg.expand_description_for_search(
    description=description,
    topic=data_topic,
)

print("\n**** Search Prompt ****\n", search_prompt)
print("\n**** Search-Focused Description ****\n", search_focused_description)

# Attach evaluator
extend_ddg.set_evaluator(GPTEvaluator(gpt4_api_key=api_key))

# Score descriptions
general_score = extend_ddg.evaluate_description(description)
search_score = extend_ddg.evaluate_description(search_focused_description)

print("\n**** Score of the General Description ****\n", general_score)
print("\n**** Score of the Search-Focused Description ****\n", search_score)
