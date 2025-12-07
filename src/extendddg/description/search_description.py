from __future__ import annotations

from typing import Any, Tuple
from beartype import beartype


@beartype
class SearchFocusedDescription:
    """Generate a search-focused expansion of an existing dataset description."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model_name
        self.temperature = float(temperature)

    def expand_description(
        self,
        description: str,
        topic: str,
    ) -> Tuple[str, str]:
        """
        Convert a human-friendly description into a search-optimized version.
        """

        prompt = (
            "Rewrite the following dataset description to improve retrieval in search engines.\n"
            f"Topic: {topic}\n\n"
            f"Description:\n{description}\n\n"
            "Provide a more detailed, keyword-rich version."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You rewrite descriptions for search indexing."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        expanded_description = response.choices[0].message.content
        return prompt, expanded_description
