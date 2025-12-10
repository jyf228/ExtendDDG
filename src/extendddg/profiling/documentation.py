import re
from typing import Any, Dict

from extendddg.parsing.documentation import DocumentationParser


class DocumentationProfiler:
    """
    Convert raw extracted documentation sections into a structured,
    standardized documentation profile suitable for LLM-based description.
    """

    IMPORTANT_FIELDS = {
        "dataset_overview": ["overview", "introduction", "about"],
        "data_collection": ["data collection", "collection", "procedures", "source"],
        "methodology": ["methodology", "methods", "study design", "approach"],
        "population": ["population", "participants", "subjects", "units"],
        "variables_description": ["variables", "fields", "attributes", "columns"],
        "limitations": ["limitations", "constraints", "weaknesses"],
        "biases": ["bias", "biases", "issues", "skew"],
        "ethics": ["ethics", "irb", "consent", "privacy"],
        "notes": ["notes", "summary", "conclusion"],
    }

    def __init__(self) -> None:
        self.documentation_parser = DocumentationParser()

    def profile(self, path: str) -> Dict[str, Any]:
        raw = self.documentation_parser.parse(path)
        breakpoint()
        return self._build_profile(raw)

    def _build_profile(self, sections: Dict[str, str]) -> Dict[str, Any]:
        profile = {}

        for output_field, keywords in self.IMPORTANT_FIELDS.items():
            profile[output_field] = self._match_section(sections, keywords)

        profile["sample_size"] = self._extract_sample_size(sections)

        return profile

    def _match_section(self, sections: Dict[str, str], keywords) -> str:
        for key in keywords:
            for section_name, text in sections.items():
                if key.lower() in section_name.lower():
                    return text.strip()
        return ""

    def _extract_sample_size(self, sections: Dict[str, str]) -> str:
        combined = " ".join(sections.values())

        patterns = [
            r"[Ss]ample size[: ]+(\d+)",
            r"[Nn]\s*=\s*(\d+)",
            r"[Ss]ample[: ]+(\d+)",
        ]

        for pattern in patterns:
            m = re.search(pattern, combined)
            if m:
                return m.group(0)

        return ""
