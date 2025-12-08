import json
from typing import Any, Tuple

from autoddg import AutoDDG, GPTEvaluator
from beartype import beartype
from pandas import DataFrame

from extendddg.evaluation.metrics import detailed_evaluate_description

from .description import DatasetDescriptionGenerator
from .profiling import CodebookProfiler, SemanticProfiler


@beartype
class ExtendDDG:
    """ExtendDDG: Extended Version of AutoDDG for Supplemental Dataset Documentation

    Args:
        client (Any): OpenAI-compatible client (e.g. ``openai.OpenAI(...)``).
        model_name (str): Default model identifier (e.g. ``"gpt-4o"``).
        description_temperature (float): Temperature for description generation.
        description_words (int): Target word count for generated descriptions.
        codebook_model_name (str | None): Override model for codebook profiling.
        semantic_model_name (str | None): Override model for semantic profiling.
        TODO: Add additional parameters.

    Examples:
        TODO: Add example usage.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        description_temperature: float = 0.0,
        description_words: int = 200,
        codebook_model_name: str | None = None,
        semantic_model_name: str | None = None,
    ):
        self.client = client
        self.model_name = model_name
        self.auto_ddg = AutoDDG(
            client=client,
            model_name=model_name,
            description_words=description_words
        )
        self.description_generator = DatasetDescriptionGenerator(
            client=client,
            model_name=model_name,
            temperature=description_temperature,
            description_words=description_words,
        )
        self.codebook_profiler = CodebookProfiler(
            client=self.client,
            model_name=codebook_model_name or model_name
        )
        self.semantic_profiler = SemanticProfiler(
            client=client,
            model_name=semantic_model_name or model_name,
        )

    def describe_dataset(
        self,
        dataset_sample: str,
        dataset_profile: str | None = None,
        use_profile: bool = False,
        semantic_profile: str | None = None,
        use_semantic_profile: bool = False,
        data_topic: str | None = None,
        use_topic: bool = False,
        # documentation_profile: str | None = None,  # TODO
        # use_documentation_profile: bool = False,  # TODO
        codebook_profile: dict[str, dict[str, str]] | None = None,
        use_codebook_profile: bool = False,
    ) -> Tuple[str, str]:
        return self.description_generator.generate_description(
            dataset_sample=dataset_sample,
            dataset_profile=dataset_profile,
            use_profile=use_profile,
            semantic_profile=semantic_profile,
            use_semantic_profile=use_semantic_profile,
            data_topic=data_topic,
            use_topic=use_topic,
            codebook_profile=json.dumps(codebook_profile) if codebook_profile else None,
            use_codebook_profile=use_codebook_profile,
        )

    def profile_dataframe(self, dataframe: DataFrame) -> Tuple[str, str]:
        return self.auto_ddg.profile_dataframe(dataframe)

    def profile_codebook(
        self, dataset_df: DataFrame, codebook_file: str
    ) -> dict[str, dict[str, Any]]:
        return self.codebook_profiler.profile_codebook(
            dataset_df=dataset_df,
            codebook_path=codebook_file,
        )

    def analyze_semantics(
        self, dataframe: DataFrame, codebook_profile: dict[str, dict[str, str]] | None
    ) -> str:
        return self.semantic_profiler.analyze_dataframe(dataframe, codebook_profile)

    def generate_topic(
        self, title: str, original_description: str | None, dataset_sample: str
    ) -> str:
        return self.auto_ddg.generate_topic(
            title=title,
            original_description=original_description,
            dataset_sample=dataset_sample,
        )

    def expand_description_for_search(self, description: str, topic: str) -> Tuple[str, str]:
        return self.auto_ddg.expand_description_for_search(description, topic)

    def evaluate_description(self, description: str) -> str:
        return self.auto_ddg.evaluate_description(description)

    def set_evaluator(self, evaluator: GPTEvaluator) -> None:
        self.auto_ddg.set_evaluator(evaluator)

    # TODO: Temp placeholder while we figure out full ExtendDDG evaluation plans
    def evaluation_metrics(self, generated_description: str, original_description: str) -> Any:
        metrics = detailed_evaluate_description(generated_description, original_description)
        return metrics
