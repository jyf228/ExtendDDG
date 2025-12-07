from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from beartype import beartype

from ..utils import load_prompts


@beartype
class DocAwareDatasetDescriptionGenerator:
    """Generate dataset descriptions that can optionally incorporate a DocumentationProfile."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        *,
        temperature: float = 0.0,
        description_words: int = 100,
    ) -> None:
        self.client = client
        self.model = model_name
        self.temperature = float(temperature)
        self.description_words = int(description_words)

        prompts = load_prompts()["dataset_description"]
        self._prompt_segments: Dict[str, str] = {
            "introduction": prompts["introduction"],
            "profile_instruction": prompts["profile_instruction"],
            "semantic_instruction": prompts["semantic_instruction"],
            "topic_instruction": prompts["topic_instruction"],
            "closing_instruction": prompts["closing_instruction"],
        }
        self._documentation_instruction = prompts.get("documentation_instruction", "")
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
    ) -> str:
        sections: Iterable[str] = [
            self._prompt_segments["introduction"].format(dataset_sample=dataset_sample)
        ]
        prompt_parts = list(sections)

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

        if use_documentation_profile and documentation_profile and self._documentation_instruction:
            prompt_parts.append(
                self._documentation_instruction.format(
                    documentation_profile=documentation_profile
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
    ) -> Tuple[str, str]:
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
