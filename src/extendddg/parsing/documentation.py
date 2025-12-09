import re
from pathlib import Path
from typing import Dict, Optional
from beartype import beartype

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


@beartype
class DocumentationParser:
    """
    Extract raw text from dataset documentation (PDF / TXT)
    and split into generic sections for profiling.
    """

    def parse(self, path: str) -> Dict[str, str]:
        """Main entry: extract text → split → filter → return dict."""
        text = self._extract(path)
        if not text:
            return {}

        sections = self._split_into_sections(text)
        relevant = self._select_relevant_sections(sections)
        return relevant

    # ---------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------
    def _extract(self, path: str) -> Optional[str]:
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(path)
        if ext in {".txt", ".md"}:
            return self._extract_text(path)

        raise ValueError(f"Unsupported documentation format: {ext}")

    def _extract_pdf(self, path: str) -> str:
        if PdfReader is None:
            raise ImportError("PyPDF2 required for PDF parsing.")

        reader = PdfReader(path)
        pages = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue

        return "\n".join(pages)

    def _extract_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---------------------------------------------------------
    # Section splitting
    # ---------------------------------------------------------
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        pattern = (
            r"(Methodology|Sampling|Survey Design|Weighting|Limitations|"
            r"Data Collection|Overview|Notes|Imputation Procedures)"
        )

        parts = re.split(pattern, text, flags=re.IGNORECASE)
        sections = {}

        if len(parts) < 3:
            sections["overview"] = text
            return sections

        for i in range(1, len(parts) - 1, 2):
            heading = parts[i].lower().strip()
            body = parts[i + 1].strip()
            sections[heading] = body

        return sections

    # ---------------------------------------------------------
    # Filtering relevant sections
    # ---------------------------------------------------------
    def _select_relevant_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        priority = [
            "methodology",
            "sampling",
            "survey design",
            "weighting",
            "limitations",
            "data collection",
            "overview",
            "notes",
        ]

        relevant = {}

        for key in priority:
            for heading, body in sections.items():
                if key in heading:
                    relevant[key] = self._shorten(body)

        if not relevant and "overview" in sections:
            relevant["overview"] = self._shorten(sections["overview"])

        return relevant

    def _shorten(self, text: str, max_len: int = 1500) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text

        cutoff = text.rfind(".", 0, max_len)
        if cutoff != -1:
            return text[:cutoff + 1]

        return text[:max_len]
