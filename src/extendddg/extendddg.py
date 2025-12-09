from typing import Any, Tuple, Optional, Dict
from autoddg import AutoDDG, GPTEvaluator
from beartype import beartype
from pandas import DataFrame
import json

# Profilers
from .profiling import SemanticProfiler
from .profiling.documentation import DocumentationProfiler


@beartype
class ExtendDDG:
    """
    ExtendDDG wraps AutoDDG and adds:
    - semantic profiling
    - documentation profiling
    """

    def __init__(self, client: Any, model_name: str, *, description_words: int = 150):
        self.client = client
        self.model_name = model_name

        # Base AutoDDG engine
        self.auto_ddg = AutoDDG(
            client=client,
            model_name=model_name,
            description_words=description_words,
        )

        # Semantic extension
        self.semantic_profiler = SemanticProfiler(
            client=client,
            model_name=model_name,
        )

        # Documentation profiling extension
        self.documentation_profiler = DocumentationProfiler()

    # ------------------------------------------------------------------
    # MAIN DESCRIPTION METHOD
    # ------------------------------------------------------------------
    def describe_dataset(
        self,
        dataset_sample: str,
        dataset_profile: Optional[str] = None,
        use_profile: bool = False,
        semantic_profile: Optional[str] = None,
        use_semantic_profile: bool = False,
        data_topic: Optional[str] = None,
        use_topic: bool = False,
        documentation_path: Optional[str] = None,
        use_documentation_profile: bool = False,   # ← YOU NEED THIS FLAG
    ) -> Tuple[str, str]:
        """
        Create dataset description using AutoDDG.
        This merges:
        - basic profile
        - semantic profile
        - documentation profile
        """

        combined_profile = dataset_profile or ""

        # Add semantic profile
        if use_semantic_profile and semantic_profile:
            combined_profile += "\n\n[Semantic Profile]\n"
            combined_profile += semantic_profile

        # Add documentation profile (ONLY IF FLAG IS TRUE)
        if use_documentation_profile and documentation_path:
            doc_profile = self.documentation_profiler.profile(documentation_path)
            combined_profile += "\n\n[Documentation Profile]\n"
            combined_profile += json.dumps(doc_profile, indent=2)

        # Call AutoDDG (it only accepts the arguments below)
        return self.auto_ddg.describe_dataset(
            dataset_sample=dataset_sample,
            dataset_profile=combined_profile,
            use_profile=True,
            data_topic=data_topic,
            use_topic=use_topic,
        )

    # ------------------------------------------------------------------
    # PASS-THROUGH HELPERS
    # ------------------------------------------------------------------
    def profile_dataframe(self, dataframe: DataFrame) -> Tuple[str, str]:
        return self.auto_ddg.profile_dataframe(dataframe)

    def analyze_semantics(self, dataframe: DataFrame, codebook=None) -> str:
        return self.semantic_profiler.analyze_dataframe(dataframe, codebook)

    def generate_topic(self, title: str, original_description: str, dataset_sample: str):
        return self.auto_ddg.generate_topic(title, original_description, dataset_sample)

    def expand_description_for_search(self, description: str, topic: str):
        return self.auto_ddg.expand_description_for_search(description, topic)

    def evaluate_description(self, description: str):
        return self.auto_ddg.evaluate_description(description)

    def set_evaluator(self, evaluator: GPTEvaluator):
        self.auto_ddg.set_evaluator(evaluator)
