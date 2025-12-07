import json
import re
from typing import Any

import pandas as pd
from pandas import DataFrame

from extendddg.profiling.codebook import CodebookProfiler

from ..utils import load_prompts


class CodebookParser:
    """Extract and parse variables table from a dataset codebook or data dictionary"""

    def __init__(self, client: Any, model_name: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model_name = model_name
        prompts = load_prompts()["codebook_parser"]
        self._template = prompts["template"]
        self._response_example = prompts["response_example"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def parse_codebook(self, dataset_df: DataFrame, codebook_path: str) -> dict[str, dict[str, Any]]:
        """Parse codebook into structured variable metadata using an LLM"""

        # Extract and preprocess the codebook
        codebook_df = self._extract_table(codebook_path)
        codebook_df = self._prepare_codebook(codebook_df, dataset_df)

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

    def _extract_table(self, codebook_path: str) -> None:
        """Extract variables table from codebook PDF or CSV file"""

        if codebook_path.endswith(".pdf"):
            return self._extract_from_pdf(codebook_path)
        elif codebook_path.endswith(".csv"):
            return self._extract_from_csv(codebook_path)
        else:
            raise ValueError(
                "Unsupported codebook format. Codebook file must either be a PDF or CSV."
            )

    def _extract_from_pdf(self, codebook_path: str) -> DataFrame:
        # TODO: Extract tables from PDF using camelot
        # https://colab.research.google.com/github/camelot-dev/camelot/blob/master/examples/camelot-quickstart-notebook.ipynb

        # TODO: Combine extracted tables into a single DataFrame if page breaks occur mid-table
        pass

    def _extract_from_csv(self, codebook_path: str) -> DataFrame:
        csv_df = pd.read_csv(codebook_path)
        return csv_df

    def _prepare_codebook(self, codebook_df: DataFrame, dataset_df: DataFrame) -> DataFrame:
        """Preprocess the codebook in preparation for prompting"""

        # Profile the codebook to infer its column structure
        codebook_profiler = CodebookProfiler(
            client=self.client,
            model_name=self.model_name,
        )
        column_mappings = codebook_profiler.analyze_codebook(codebook_df)

        # Standardize column names and drop less relevant columns
        codebook_df = self._standardize_df_column_names(codebook_df, column_mappings)

        # Drop duplicate rows based on variable_name
        codebook_df = codebook_df.drop_duplicates(subset=['variable_name'], keep='first')

        # Include only variables that were included in the dataset DataFrame
        # This ensures that if the dataset was sampled, we're only keeping relevant variables
        dataset_columns = set(dataset_df.columns)
        codebook_df = codebook_df[codebook_df['variable_name'].isin(dataset_columns)]

        return codebook_df

    def _standardize_df_column_names(self, df: DataFrame, col_mappings: dict[str, int]) -> DataFrame:
        """Standardize DataFrame columns based on a mapping of standard column names to the column indices"""
        rename_mapping = {}
        columns_to_keep = []

        for key, col_index in col_mappings.items():
            if key in ['variable_name', 'description']:
                if isinstance(col_index, int):
                    curr_col_name = df.columns[col_index]
                    rename_mapping[curr_col_name] = key
                    columns_to_keep.append(key)

        # Rename columns to the standardized names and keep only the renamed columns
        df = df.rename(columns=rename_mapping)
        return df[columns_to_keep]

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
