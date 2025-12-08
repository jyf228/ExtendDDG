import json
import re
from typing import Any

from beartype import beartype
from pandas import DataFrame

from ..parsing import CodebookParser
from ..utils import load_prompts


@beartype
class CodebookProfiler:
    """Profile a dataset codebook or data dictionary using an LLM"""

    def __init__(self, client: Any, model_name: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model_name = model_name
        self.codebook_parser = CodebookParser(
            client=self.client,
            model_name=self.model_name
        )
        prompts = load_prompts()["codebook_profiler"]
        self._template = prompts["template"]
        self._response_example = prompts["response_example"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def profile_codebook(
        self, dataset_df: DataFrame, codebook_path: str
    ) -> dict[str, dict[str, Any]]:
        """Profile codebook and build a structured dictionary of variables using an LLM"""

        # Extract and parse the codebook into a standardized DataFrame
        codebook_df = self.codebook_parser.parse_codebook(codebook_path, dataset_df)

        prompt = self._build_prompt(codebook_df)
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

    def _build_prompt(self, codebook_df: DataFrame) -> str:
        codebook_json = codebook_df.to_dict(orient="records")
        prompt =  self._user_prompt.format(
            template=self._template,
            response_example=self._response_example,
            codebook_table=codebook_json,
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
