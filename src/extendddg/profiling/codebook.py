import json
import re
from typing import Any

from pandas import DataFrame

from ..utils import load_prompts


# @beartype
class CodebookProfiler:
    """Infer structure from a dataset codebook or data dictionary using an LLM"""

    def __init__(self, client: Any, model_name: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model_name = model_name
        prompts = load_prompts()["codebook_profiler"]
        self._template = prompts["template"]
        self._response_example = prompts["response_example"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def analyze_codebook(self, codebook_df: DataFrame) -> str:
        """
        Map codebook columns to standard roles

        Args:
            codebook_df: Input codebook DataFrame
        """

        def _get_sample(data_pd: DataFrame, sample_size: int) -> DataFrame:
            if sample_size < len(data_pd):
                return data_pd.sample(sample_size, random_state=9)
            return data_pd

        # Sample the codebook
        codebook_sample = _get_sample(codebook_df, 10)

        # Infer the structure of the codebook from the sample
        prompt = self._build_prompt(codebook_sample)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        response_text = self._fix_json_response(response_text)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None

    def _build_prompt(self, codebook_sample: DataFrame) -> str:
        codebook_json = codebook_sample.to_dict(orient="records")
        prompt =  self._user_prompt.format(
            template=self._template,
            response_example=self._response_example,
            codebook_sample=codebook_json,
        )
        return prompt

    def _fix_json_response(self, response_text: str) -> str:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return response_text

        response_body = match.group()
        open_braces = response_body.count("{")
        close_braces = response_body.count("}")
        response_body += "}" * (open_braces - close_braces)
        response_body = re.sub(r",\s*}", "}", response_body)
        return response_body
