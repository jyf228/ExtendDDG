from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from beartype import beartype

from ..utils import load_prompts


@beartype
class DatasetDescriptionGenerator:
    """Generate human-readable descriptions for tabular datasets"""

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.0,
        description_words: int = 200,
    ) -> None:
        self.client = client
        self.model = model_name
        self.temperature = float(temperature)
        self.description_words = int(description_words)
        prompts = load_prompts()["dataset_description"]
        self._prompt_segments: Dict[str, str] = {
            "introduction": prompts["introduction"],
            "dataset_sample_instruction": prompts["dataset_sample_instruction"],
            "profile_instruction": prompts["profile_instruction"],
            "semantic_instruction": prompts["semantic_instruction"],
            "topic_instruction": prompts["topic_instruction"],
            # "documentation_instruction": prompts["documentation_instruction"],
            "documentation_instruction": prompts.get(
                "documentation_instruction",
                "\nThe following external documentation may help:\n{documentation_profile}\n"   # TODO
            ),
            "codebook_instruction": prompts["codebook_instruction"],
            "closing_instruction": prompts["closing_instruction"],
        }
        self._system_message = prompts["system_message"].strip()

    def _generate_prompt(
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
        codebook_profile: str | None = None,
        use_codebook_profile: bool = False,
    ) -> str:
        sections: Iterable[str] = [self._prompt_segments["introduction"]]
        prompt_parts = list(sections)

        # Only include the dataset sample if no codebook profile is provided since
        # the codebook profile already describes the dataset structure at a higher-level.
        # This limits token usage by reducing redundancy in the prompt.
        if not use_codebook_profile:
            prompt_parts.append(
                self._prompt_segments["dataset_sample_instruction"].format(
                    dataset_sample=dataset_sample
                )
            )

        if use_profile and dataset_profile:
            prompt_parts.append(
                self._prompt_segments["profile_instruction"].format(dataset_profile=dataset_profile)
            )

        if use_semantic_profile and semantic_profile:
            prompt_parts.append(
                self._prompt_segments["semantic_instruction"].format(
                    semantic_profile=semantic_profile
                )
            )

        if use_topic and data_topic:
            prompt_parts.append(
                self._prompt_segments["topic_instruction"].format(data_topic=data_topic)
            )

        if use_documentation_profile and documentation_profile:
            prompt_parts.append(
                self._prompt_segments["documentation_instruction"].format(
                    documentation_profile=documentation_profile
                )
            )

        if use_codebook_profile and codebook_profile:
            prompt_parts.append(
                self._prompt_segments["codebook_instruction"].format(
                    codebook_profile=codebook_profile
                )
            )

        prompt_parts.extend(
            [
                self._prompt_segments["closing_instruction"],
                f"Target length: approximately {self.description_words} words.",
            ]
        )

        return "\n".join(prompt_parts)

    def generate_description(
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
        codebook_profile: str | None = None,
        use_codebook_profile: bool = False,
    ) -> Tuple[str, str]:
        """
        Call the model and return prompt and description

        Args:
            dataset_sample: CSV text sample
            dataset_profile: Structural profile
            use_profile: Include profile if True
            semantic_profile: Semantic profile
            use_semantic_profile: Include semantic profile if True
            data_topic: Short topic string
            use_topic: Include topic if True
            codebook_profile: Codebook profile
            use_codebook_profile: Include codebook profile if True

        Returns:
            (prompt, description)
        """

        prompt = self._generate_prompt(
            dataset_sample=dataset_sample,
            dataset_profile=dataset_profile,
            use_profile=use_profile,
            semantic_profile=semantic_profile,
            use_semantic_profile=use_semantic_profile,
            data_topic=data_topic,
            use_topic=use_topic,
            documentation_profile=documentation_profile,
            use_documentation_profile=use_documentation_profile,
            codebook_profile=codebook_profile,
            use_codebook_profile=use_codebook_profile,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        description = response.choices[0].message.content
        return prompt, description
