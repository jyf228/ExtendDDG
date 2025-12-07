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
    Parse supplemental documentation (PDF or text files)
    and produce a compact DocumentationProfile.
    """

    def __init__(self) -> None:
        pass

    def parse(self, path: str) -> Dict[str, str]:
        """Main entry point for extracting and summarizing documentation."""
        text = self._extract(path)
        if not text:
            return {}

        sections = self._split_into_sections(text)
        relevant = self._select_relevant_sections(sections)
        profile = self._build_profile(relevant)
        return profile

    # --------------------------
    # Extraction helpers
    # --------------------------
    def _extract(self, path: str) -> Optional[str]:
        """Detect file type and run the appropriate extraction method."""
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(path)
        if ext in [".txt", ".md"]:
            return self._extract_text(path)

        raise ValueError(f"Unsupported documentation format: {ext}")

    def _extract_pdf(self, path: str) -> str:
        """Simple PDF text extraction."""
        if PdfReader is None:
            raise ImportError("PyPDF2 must be installed for PDF extraction.")

        reader = PdfReader(path)
        pages = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue

        return "\n".join(pages)

    def _extract_text(self, path: str) -> str:
        """Read plain text files."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # --------------------------
    # Section splitting
    # --------------------------
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split by common documentation headings.
        This helps isolate the useful parts like methodology or sampling.
        """
        pattern = r"(Methodology|Sampling|Weighting|Limitations|Notes|Overview|Data Collection|Imputation Procedures|Survey Design)"
        matches = re.split(pattern, text, flags=re.IGNORECASE)

        sections = {}

        if len(matches) < 3:
            sections["full_document"] = text
            return sections

        for i in range(1, len(matches) - 1, 2):
            heading = matches[i].strip().lower()
            body = matches[i + 1].strip()
            sections[heading] = body

        return sections

    # --------------------------
    # Relevance filtering
    # --------------------------
    def _select_relevant_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Keep only the sections most important for dataset description."""
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
                if key in heading and heading not in relevant:
                    relevant[heading] = self._shorten(body)

        # If no headings matched, fall back to full document
        if not relevant and "full_document" in sections:
            relevant["overview"] = self._shorten(sections["full_document"])

        return relevant

    def _shorten(self, text: str, max_len: int = 1200) -> str:
        """Trim long sections to avoid token limit issues."""
        text = text.strip()
        if len(text) <= max_len:
            return text

        cutoff = text.rfind(".", 0, max_len)
        if cutoff != -1:
            return text[:cutoff + 1]

        return text[:max_len]

    # --------------------------
    # DocumentationProfile builder
    # --------------------------
    def _build_profile(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Return a compact, structured documentation summary."""
        profile = {
            "methodology": sections.get("methodology", ""),
            "sampling": sections.get("sampling", ""),
            "weighting": sections.get("weighting", ""),
            "limitations": sections.get("limitations", ""),
            "survey_design": sections.get("survey design", ""),
            "data_collection": sections.get("data collection", ""),
            "notes": sections.get("notes", ""),
            "overview": sections.get("overview", ""),
        }

        # Remove empty entries
        return {k: v for k, v in profile.items() if v}
