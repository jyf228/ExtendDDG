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
documentation_file = None # Set to None if not available

# Set dataset title and original dataset description
title = "2023-24 Religious Landscape Study (RLS) Dataset"
original_description = (
    "This Pew Research Center survey was conducted among a nationally representative sample of adults to provide estimates of the U.S. population’s religious composition, beliefs and practices.\n "
    "Data access and use\n"
    "Pew Research Center is releasing two versions of the dataset – a public-use file (PUF) and restricted-use file (RUF). Both datasets include information on all 36,908 of the survey’s respondents. "
    "The PUF does not include any information about geography, and it excludes information on several other sensitive variables (including detailed variables about religious identity). These geographic "
    "and other sensitive variables will be included only in the RUF, which we intend to make accessible at a future date via ICPSR with a data use agreement. "
    "Refer to the enclosed readme file for additional details.\nTopics\n"
    "The survey covers topics such as religious identity, religious beliefs and practices, spirituality, social and political values, and more.\n\n"
    "Sample design\n"
    "The survey is designed to be representative of the U.S. adult population, and of all 50 states and the District of Columbia. We used address-based sampling (ABS) and mailed invitation "
    "letters to randomly sampled addresses from the United States Postal Service’s Computerized Delivery Sequence File. This approach gave nearly all U.S. adults a chance of being selected to "
    "participate in the survey. People who received our invitation had the option of completing the survey online, on paper, or by calling a toll-free number and completing the survey by telephone "
    "with an interviewer. The survey was conducted in two languages, English and Spanish. Responses were collected from July 17, 2023, to March 4, 2024."
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
print("\n**** User-Focused Description ****\n", description)

# ---------------------------------------------------------
# 9. Search-focused description
# ---------------------------------------------------------
search_prompt, search_focused_description = extend_ddg.expand_description_for_search(
    description=description,
    topic=data_topic,
)
print("\n**** Search-Focused Description ****\n", search_focused_description)

# ---------------------------------------------------------
# 10. Evaluation
# ---------------------------------------------------------
extend_ddg.set_evaluator(GPTEvaluator(gpt4_api_key=api_key))

general_score = extend_ddg.evaluate_description(description)
search_score = extend_ddg.evaluate_description(search_focused_description)

print("\n**** Score of the General Description ****\n", general_score)
print("\n**** Score of the Search-Focused Description ****\n", search_score)
