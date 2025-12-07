from typing import Any, Tuple

from autoddg import AutoDDG, GPTEvaluator
from beartype import beartype
from pandas import DataFrame

from .description import DocAwareDatasetDescriptionGenerator
from .profiling import DocumentationProfiler, SemanticProfiler


@beartype
class ExtendDDG:
    """ExtendDDG: Extended Version of AutoDDG for Supplemental Dataset Documentation"""

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        description_words: int = 150,
        description_temperature: float = 0.0,
    ):
        self.client = client
        self.model_name = model_name
        self.auto_ddg = AutoDDG(
            client=client,
            model_name=model_name,
            description_words=description_words,
            description_temperature=description_temperature,
        )
        self.description_generator = DocAwareDatasetDescriptionGenerator(
            client=client,
            model_name=model_name,
            description_words=description_words,
            temperature=description_temperature,
        )
        self.semantic_profiler = SemanticProfiler(
            client=client,
            model_name=model_name,
        )
        self.documentation_profiler = DocumentationProfiler(
            client=client,
            model_name=model_name,
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
        documentation_profile: str | None = None,
        use_documentation_profile: bool = False,
    ) -> Tuple[str, str]:
        return self.description_generator.generate_description(
            dataset_sample=dataset_sample,
            dataset_profile=dataset_profile,
            use_profile=use_profile,
            semantic_profile=semantic_profile,
            use_semantic_profile=use_semantic_profile,
            data_topic=data_topic,
            use_topic=use_topic,
            documentation_profile=documentation_profile,
            use_documentation_profile=use_documentation_profile,
        )

    def profile_dataframe(self, dataframe: DataFrame) -> Tuple[str, str]:
        return self.auto_ddg.profile_dataframe(dataframe)

    def analyze_semantics(self, dataframe: DataFrame, codebook: DataFrame | None = None) -> str:
        return self.semantic_profiler.analyze_dataframe(dataframe, codebook)

    def build_documentation_profile(
        self,
        *,
        documentation_path: str | None = None,
        documentation_text: str | None = None,
    ) -> str:
        if documentation_path:
            return self.documentation_profiler.build_profile_from_path(documentation_path)
        if documentation_text:
            return self.documentation_profiler.build_profile_from_text(documentation_text)
        raise ValueError("Provide either documentation_path or documentation_text.")

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
