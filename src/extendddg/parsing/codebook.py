import pandas as pd
from beartype import beartype
from pandas import DataFrame


@beartype
class CodebookParser:
    """Extract and parse variables table from a dataset codebook or data dictionary"""

    def __init__(self) -> None:
        pass

    def extract_variables(self, codebook_path: str) -> None:
        """Extract variables table from codebook PDF or CSV file"""

        if codebook_path.endswith(".pdf"):
            self._extract_from_pdf(codebook_path)
        elif codebook_path.endswith(".csv"):
            self._extract_from_csv(codebook_path)
        else:
            raise ValueError("Unsupported codebook format. Codebook file must either be a PDF or CSV.")

    def _extract_from_pdf(self, codebook_path: str) -> DataFrame:
        # TODO: Extract tables from PDF using camelot
        # https://colab.research.google.com/github/camelot-dev/camelot/blob/master/examples/camelot-quickstart-notebook.ipynb

        # TODO: Combine extracted tables into a single DataFrame if page breaks occur mid-table
        pass

    def _extract_from_csv(self, codebook_path: str) -> DataFrame:
        csv_df = pd.read_csv(codebook_path)
        return csv_df
