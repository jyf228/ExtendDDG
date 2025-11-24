import pandas as pd
from openai import OpenAI

from extendddg import ExtendDDG, GPTEvaluator
from extendddg.utils import get_sample

# Setup OpenAI client
api_key = "your-api-key"
client = OpenAI(api_key=api_key)
model_name = "gpt-4o-mini"

# Initialize ExtendDDG
extend_ddg = ExtendDDG(client=client, model_name=model_name)

# Load dataset
# TODO: Replace with our own example dataset
csv_file = "clark_dataset.csv"
title = "Renal Cell Carcinoma"
original_description = (
    "This study reports a large-scale proteogenomic analysis of ccRCC to discern the functional impact "
    "of genomic alterations and provides evidence for rational treatment selection stemming from ccRCC pathobiology"
)
csv_df = pd.read_csv(csv_file)

# Sample rows
sample_df, dataset_sample = get_sample(csv_df, sample_size=100)

# TODO: Add handling for supplemental dataset information

# Generate profiles
basic_profile, structural_profile = extend_ddg.profile_dataframe(csv_df)
print("**** Basic Profile ****\n", basic_profile)
print("\n**** Structural Profile ****\n", structural_profile)

semantic_profile_details = extend_ddg.analyze_semantics(sample_df)
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

# General description
prompt, description = extend_ddg.describe_dataset(
    dataset_sample=dataset_sample,
    dataset_profile=basic_profile,
    use_profile=True,
    semantic_profile=semantic_profile,
    use_semantic_profile=True,
    data_topic=data_topic,
    use_topic=True,
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
