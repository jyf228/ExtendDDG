import json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from bert_score import score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def detailed_evaluate_description(
    generated_description: str, ground_truth: str
) -> dict[str, float]:
    bert_score = _compute_bert(generated_description, ground_truth)
    rouge_score = _compute_rouge(generated_description, ground_truth)
    bleu_score = _compute_bleu(generated_description, ground_truth)

    return bert_score | rouge_score | {"BLEU": bleu_score}


def _compute_bert(
    generated_description: str, ground_truth: str, model="bert-base-uncased"
) -> dict[str, float]:
    """
    Evaluate the generated description using BERTScore.
    """

    P, R, F1 = score(
        [generated_description],
        [ground_truth],
        lang="en",
        model_type=model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return {"precision": float(P.item()), "recall": float(R.item()), "f1": float(F1.item())}


def _compute_rouge(generated_description: str, ground_truth: str) -> dict[str, float]:
    """
    Evaluate the generated description using ROUGE.
    """

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = scorer.score(ground_truth, generated_description)

    result = {}
    result["ROUGE-1:"] = scores["rouge1"].fmeasure
    result["ROUGE-2:"] = scores["rouge2"].fmeasure
    result["ROUGE-L:"] = scores["rougeL"].fmeasure

    return result


def _compute_bleu(generated_description: str, ground_truth: str) -> float:
    """
    Evaluate the generated description using BLEU score.
    """

    reference_tokens = ground_truth.split()
    candidate_tokens = generated_description.split()

    smooth = SmoothingFunction().method1

    score_val = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)

    return score_val


def evaluate_variable_alignment(
    description: str,
    *,
    codebook: str | pd.DataFrame,
    client: Any,
    model_name: str = "gpt-4o-mini",
    sample_size: int = 10,
    random_state: int = 9,
) -> Dict[str, Any]:
    """
    Check whether variable meanings/units from a codebook are reflected correctly
    in a generated description. Uses an LLM grader per sampled variable.
    """

    codebook_df = _load_codebook(codebook)
    if codebook_df.empty:
        raise ValueError("Codebook is empty; cannot evaluate variable alignment.")

    sample_df = codebook_df.sample(min(sample_size, len(codebook_df)), random_state=random_state)
    results: List[Dict[str, Any]] = []
    supported_count = 0

    for row in sample_df.itertuples(index=False):
        var_name, label, values, value_labels = _extract_codebook_fields(row)
        grading_prompt = _build_variable_prompt(description, var_name, label, values, value_labels)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are grading whether a dataset description correctly reflects a codebook variable. "
                        "Return a JSON object only."
                    ),
                },
                {"role": "user", "content": grading_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        parsed = _safe_json_parse(content)

        supported = parsed.get("supported", "").lower()
        if supported in {"yes", "true"}:
            supported_count += 1
        results.append(
            {
                "variable": var_name,
                "label": label,
                "supported": supported,
                "rationale": parsed.get("rationale", ""),
            }
        )

    support_rate = supported_count / len(results) if results else 0.0
    return {"support_rate": support_rate, "results": results}


def _load_codebook(codebook: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(codebook, pd.DataFrame):
        return codebook
    return pd.read_csv(codebook)


def _extract_codebook_fields(row: Tuple[Any, ...]) -> Tuple[str, str, str, str]:
    row_dict = row._asdict() if hasattr(row, "_asdict") else {}
    var_name = str(row_dict.get("Variable", row[0]))
    label = str(row_dict.get("Variable_Label", row_dict.get("Label", "")))
    values = str(row_dict.get("Values", ""))
    value_labels = str(row_dict.get("Value_Labels", ""))
    return var_name, label, values, value_labels


def _build_variable_prompt(
    description: str, var_name: str, label: str, values: str, value_labels: str
) -> str:
    return (
        "Dataset description:\n"
        f"{description}\n\n"
        "Codebook entry:\n"
        f"Variable: {var_name}\n"
        f"Label: {label}\n"
        f"Values: {values}\n"
        f"Value Labels: {value_labels}\n\n"
        "Question: Does the description correctly capture this variable's meaning or units? "
        "Respond with JSON: {\"variable\": \"...\", \"supported\": \"yes\"|\"no\"|\"maybe\", \"rationale\": \"...\"}."
        "Be concise and base your judgment strictly on the provided description."
    )


def _safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"supported": "maybe", "rationale": f"Could not parse model response: {text[:200]}"}


# ---------------------------- Additional evaluation helpers ----------------------------
def semantic_consistency_check(
    description: str,
    codebook: pd.DataFrame,
    embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Compare embeddings of description sentences against codebook variables; flag low-similarity items.

    Args:
        description: Generated dataset description.
        codebook: DataFrame with variable_name and description/label columns.
        embedder: Callable that returns list of embedding vectors given list of texts.
        threshold: Cosine similarity cutoff below which variables are flagged.
    """
    sentences = _split_sentences(description)
    if not sentences:
        return {"average_similarity": 0.0, "flagged": [], "per_variable": []}

    var_texts = []
    for row in codebook.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else {}
        var_name = str(row_dict.get("variable_name", row_dict.get("Variable", row[0])))
        label = str(row_dict.get("description", row_dict.get("Variable_Label", row_dict.get("Label", ""))))
        full_text = f"{var_name}: {label}".strip()
        var_texts.append((var_name, full_text))

    sent_embs = torch.tensor(embedder(sentences), dtype=torch.float32)
    var_embs = torch.tensor(embedder([v[1] for v in var_texts]), dtype=torch.float32)

    # Compute cosine similarity matrix
    sent_norm = torch.nn.functional.normalize(sent_embs, dim=1)
    var_norm = torch.nn.functional.normalize(var_embs, dim=1)
    sim_matrix = torch.matmul(var_norm, sent_norm.T)

    per_variable = []
    flagged = []
    sims = []
    for i, (var_name, _) in enumerate(var_texts):
        max_sim = float(sim_matrix[i].max().item())
        sims.append(max_sim)
        entry = {"variable": var_name, "max_similarity": max_sim}
        per_variable.append(entry)
        if max_sim < threshold:
            flagged.append(entry)

    avg_sim = float(sum(sims) / len(sims)) if sims else 0.0
    return {
        "average_similarity": avg_sim,
        "flagged": flagged,
        "per_variable": per_variable,
    }


def length_normalized_completeness(raw_score: float, description: str, words_per_unit: int = 100) -> float:
    """
    Normalize a completeness score by description length to reduce length inflation.

    Args:
        raw_score: Original completeness score (e.g., 0-10 scale).
        description: Generated description text.
        words_per_unit: Word count that should correspond to one unit of completeness.
    """
    words = max(len(description.split()), 1)
    factor = max(words / float(words_per_unit), 1.0)
    return raw_score / factor


def summarize_variable_support(
    results: List[Dict[str, Any]],
    domain_map: Dict[str, str] | None = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Summarize variable-level support results with domain breakdown and top unsupported variables.

    Args:
        results: Output list from evaluate_variable_alignment.
        domain_map: Optional mapping from variable name to domain/category.
        top_n: Number of unsupported variables to surface.
    """
    support_counts = {"yes": 0, "no": 0, "maybe": 0}
    domain_counts: Dict[str, Dict[str, int]] = {}
    unsupported: List[Dict[str, Any]] = []

    for entry in results:
        supported = str(entry.get("supported", "")).lower()
        support_counts[supported] = support_counts.get(supported, 0) + 1
        var_name = entry.get("variable", "")
        domain = (domain_map or {}).get(var_name, "unknown")
        domain_counts.setdefault(domain, {"yes": 0, "no": 0, "maybe": 0})
        domain_counts[domain][supported] = domain_counts[domain].get(supported, 0) + 1
        if supported == "no":
            unsupported.append(entry)

    support_total = sum(support_counts.values())
    support_rate = support_counts.get("yes", 0) / support_total if support_total else 0.0
    top_unsupported = unsupported[:top_n]

    return {
        "support_rate": support_rate,
        "support_counts": support_counts,
        "domain_counts": domain_counts,
        "top_unsupported": top_unsupported,
    }


def coverage_of_key_dimensions(description: str) -> Dict[str, Any]:
    """
    Heuristic coverage check for critical metadata dimensions.
    Returns coverage booleans and matched keywords per dimension.
    """
    dimensions = {
        "population": ["adults", "children", "households", "respondents", "population", "sample"],
        "time_period": ["year", "wave", "cohort", "period", "date", "201", "202", "199"],
        "geography": ["state", "county", "cz", "zip", "region", "country", "city"],
        "methodology_sampling": ["survey", "sample", "sampling", "weight", "imputation", "response rate"],
        "weighting_imputation": ["weight", "weighted", "imputation", "impute", "replicate", "repwt"],
        "variable_definitions": ["variable", "codebook", "label", "definition", "units", "scale"],
    }

    desc_lower = description.lower()
    coverage = {}
    for dim, keywords in dimensions.items():
        hits = [kw for kw in keywords if kw.lower() in desc_lower]
        coverage[dim] = {"covered": bool(hits), "hits": hits}

    covered_count = sum(1 for v in coverage.values() if v["covered"])
    coverage_score = covered_count / len(dimensions) if dimensions else 0.0
    return {"coverage": coverage, "coverage_score": coverage_score}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# ---------------------------- New metrics for ExtendDDG vs. AutoDDG comparisons ----------------------------
def redundancy_verbosity_metrics(description: str) -> Dict[str, Any]:
    """
    Measure repetition and verbosity without needing a reference description.

    Returns:
        repeated_fact_rate: Share of sentences that appear more than once (after normalization).
        bigram_redundancy: Share of bigrams that are repeated, to catch phrasing loops.
        avg_sentence_length: Average tokens per sentence, a proxy for verbosity.
    """
    sentences = [re.sub(r"\s+", " ", s.strip().lower()) for s in _split_sentences(description)]
    if not sentences:
        return {"repeated_fact_rate": 0.0, "bigram_redundancy": 0.0, "avg_sentence_length": 0.0}

    counts = Counter(sentences)
    repeated = sum(1 for c in counts.values() if c > 1)
    repeated_fact_rate = repeated / len(counts)

    tokens = re.findall(r"\w+", description.lower())
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)
    repeated_bigram_total = sum(c for c in bigram_counts.values() if c > 1)
    bigram_redundancy = repeated_bigram_total / max(len(bigrams), 1)

    avg_sentence_length = len(tokens) / len(sentences)
    return {
        "repeated_fact_rate": repeated_fact_rate,
        "bigram_redundancy": bigram_redundancy,
        "avg_sentence_length": avg_sentence_length,
    }


def unique_fact_ratio(description: str) -> float:
    """
    Ratio of unique sentences to total sentences; higher implies less redundancy.
    """
    sentences = [re.sub(r"\s+", " ", s.strip().lower()) for s in _split_sentences(description)]
    if not sentences:
        return 0.0
    return len(set(sentences)) / len(sentences)


def helpfulness_usefulness_score(
    description: str,
    *,
    intended_use: str,
    client: Any,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    LLM-judged usefulness for the intended audience/task, separate from fluency.

    Returns usefulness_score (1-5) and a short rationale grounded in the text.
    """
    prompt = (
        "You are grading how useful a dataset description is for the stated audience/task.\n"
        f"Intended use: {intended_use}\n"
        f"Description:\n{description}\n\n"
        "Rate usefulness 1-5 where 5 = directly actionable, specific, and answers likely reader questions; "
        "1 = unhelpful, generic, or missing key details.\n"
        "Respond ONLY with JSON: {\"usefulness_score\": 1-5, \"rationale\": \"...snippet-backed rationale...\"}"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "Be concise and return JSON only."}, {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    parsed = _safe_json_parse(response.choices[0].message.content)
    score_val = int(parsed.get("usefulness_score", 0)) if str(parsed.get("usefulness_score", "")).isdigit() else 0
    return {"usefulness_score": score_val, "rationale": parsed.get("rationale", "")}


def specificity_vagueness_score(
    description: str,
    *,
    client: Any,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    LLM rubric for specificity vs. vagueness (1-5), with snippet-backed justification.
    """
    prompt = (
        "Rate the description 1-5 on specificity (5 = concrete, detailed, explicit entities/attributes/actions; "
        "1 = vague, generic, hand-wavey). Provide a one-sentence justification citing a snippet.\n"
        f"Description:\n{description}\n"
        "Respond ONLY with JSON: {\"specificity_score\": 1-5, \"justification\": \"...\"}"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "Be concise and return JSON only."}, {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    parsed = _safe_json_parse(response.choices[0].message.content)
    score_val = int(parsed.get("specificity_score", 0)) if str(parsed.get("specificity_score", "")).isdigit() else 0
    return {"specificity_score": score_val, "justification": parsed.get("justification", "")}


def semantic_quality_composite(
    generated_description: str,
    reference_description: str,
    *,
    embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
) -> Dict[str, float]:
    """
    Composite semantic quality: combines BERTScore F1 with embedding cosine between pooled representations.
    The cosine term penalizes semantic drift not captured by token overlap.
    """
    bert = _compute_bert(generated_description, reference_description)
    bert_f1 = bert["f1"]

    cosine = 0.0
    if embedder:
        gen_vec = torch.tensor(embedder([generated_description])[0], dtype=torch.float32)
        ref_vec = torch.tensor(embedder([reference_description])[0], dtype=torch.float32)
        gen_norm = torch.nn.functional.normalize(gen_vec, dim=0)
        ref_norm = torch.nn.functional.normalize(ref_vec, dim=0)
        cosine = float(torch.dot(gen_norm, ref_norm).item())

    composite = (bert_f1 + cosine) / 2 if embedder else bert_f1
    return {"bert_f1": bert_f1, "cosine_similarity": cosine, "composite_quality": composite}
