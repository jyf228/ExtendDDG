import json
import re
import warnings
from typing import Any

import camelot
import pandas as pd
from beartype import beartype
from pandas import DataFrame
from pypdf import PdfReader

from ..utils import load_prompts

warnings.filterwarnings('ignore', message='Cannot close object; pdfium library is destroyed. This may cause a memory leak.')


PARSING_METHODS = ['lattice', 'stream', 'network', 'hybrid']
PROCESS_BACKGROUND_OPTIONS = [False, True]

@beartype
class CodebookParser:
    """Parse a dataset codebook or data dictionary from a PDF or CSV file into a standardized DataFrame"""

    def __init__(self, client: Any, model_name: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model_name = model_name
        prompts = load_prompts()["codebook_parser"]
        self._template = prompts["template"]
        self._response_example = prompts["response_example"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def parse_codebook(self, codebook_path: str, dataset_df: DataFrame) -> DataFrame:
        """Extract and process the codebook in preparation for profiling"""

        # Step 1: Extract the codebook table from the file into a DataFrame
        codebook_df = self._extract_table(codebook_path)

        # Step 2: Infer the codebook structure and get column mappings using an LLM
        column_mappings = self._analyze_codebook_structure(codebook_df)

        # Step 3: Standardize column names and drop less relevant columns
        codebook_df = self._standardize_df_column_names(codebook_df, column_mappings)

        # Step 4: Drop duplicate rows based on variable_name
        codebook_df = codebook_df.drop_duplicates(subset=['variable_name'], keep='first')

        # Step 5: Include only variables that were included in the dataset DataFrame
        # This ensures that if the dataset was sampled, we're only keeping relevant variables
        dataset_columns = set(dataset_df.columns)
        codebook_df = codebook_df[codebook_df['variable_name'].isin(dataset_columns)]

        return codebook_df

    def _extract_table(self, codebook_path: str) -> DataFrame:
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
        """Read and process tables from a codebook PDF file into a DataFrame"""

        # Verify file can be opened before processing
        try:
            reader = PdfReader(codebook_path)
            if len(reader.pages) == 0:
                raise ValueError(f"No pages found in {codebook_path}")
        except Exception as e:
            raise IOError(f"Failed to read {codebook_path}: {e}") from e

        # Read tables from the PDF
        tables = self._read_tables_from_pdf(codebook_path)

        # Clean tables by replacing new line characters with spaces
        for i in range(len(tables)):
            tables[i] = tables[i].replace(r'\n', ' ', regex=True)

        # Merge tables separated by page breaks into one DataFrame
        # For now, we assume all tables extracted from one file are part of the same codebook
        combined_df = pd.concat(tables, ignore_index=True)
        return combined_df

    def _read_tables_from_pdf(self, codebook_path: str) -> list[DataFrame]:
        """Attempt to read tables from a PDF using multiple parsing methods"""

        # Attempt to read tables with the specified parsing method and process_background option
        def _attempt_read(
            parsing_method: str, process_background: bool = False
        ) -> list[DataFrame]:
            try:
                # Build kwargs conditionally
                kwargs = {
                    'flavor': parsing_method,
                }
                # Add process_background condtionally since only 'lattice' parsing supports it
                if parsing_method == 'lattice' and process_background:
                    kwargs['process_background'] = process_background

                tables = camelot.read_pdf(codebook_path, pages='all', **kwargs)

                if len(tables) == 0:
                    return []
                return [table.df for table in tables]
            except Exception as e:
                raise IOError(f"Failed to read tables from {codebook_path}: {e}") from e

        # Fallback logic to try different parameter options for reading the PDF tables
        for parsing_method in PARSING_METHODS:
            if parsing_method == 'lattice':
                for pb_option in PROCESS_BACKGROUND_OPTIONS:
                    if not _attempt_read(parsing_method, pb_option):
                        # Try reading with another process_background option
                        continue
                    return _attempt_read(parsing_method, pb_option)
            else:
                if not _attempt_read(parsing_method):
                    # Try reading with another parsing method
                    continue
                return _attempt_read(parsing_method)

        # If no tables were successfully found after fallback options were attempted, raise an error
        raise ValueError(
            f"No tables detected in {codebook_path}. Codebook must have "
            "at least one table containing variable information."
        )

    def _extract_from_csv(self, codebook_path: str) -> DataFrame:
        csv_df = pd.read_csv(codebook_path)
        return csv_df

    def _analyze_codebook_structure(self, codebook_df: DataFrame) -> dict[str, Any]:
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

    def _standardize_df_column_names(
        self, df: DataFrame, col_mappings: dict[str, int]
    ) -> DataFrame:
        """Standardize DataFrame columns based on a mapping of standard column names to the column indices"""
        rename_mapping = {}
        columns_to_keep = []

        for key, col_index in col_mappings.items():
            if key in ['variable_name', 'description', 'variable_type']:
                if isinstance(col_index, int):
                    curr_col_name = df.columns[col_index]
                    rename_mapping[curr_col_name] = key
                    columns_to_keep.append(key)

        # Rename columns to the standardized names and keep only the renamed columns
        df = df.rename(columns=rename_mapping)
        return df[columns_to_keep]

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
