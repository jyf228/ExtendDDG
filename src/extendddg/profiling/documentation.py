from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from beartype import beartype
from pypdf import PdfReader

from ..utils import load_prompts


@dataclass
class DocumentationChunk:
    text: str
    page: int | None = None
    heading: str | None = None
    score: float = 0.0


@beartype
class DocumentationProfiler:
    """
    Build a succinct documentation profile from supplemental docs (PDF or text).

    The profiler:
    1) extracts text,
    2) chunks by headings/size,
    3) scores chunks using structural and keyword priors, and
    4) summarizes the top chunks into a structured DocumentationProfile with an LLM.
    """

    # Lightweight lexical cues for relevance scoring
    _KEYWORDS: Sequence[str] = (
        "methodology",
        "methods",
        "data collection",
        "sampling",
        "sample",
        "weight",
        "imputation",
        "missing",
        "bias",
        "privacy",
        "anonym",
        "license",
        "open",
        "usage",
        "caution",
        "variable",
        "codebook",
        "population",
        "frame",
        "survey",
        "instrument",
        "limitation",
        "update",
    )

    _HEADING_PRIORS: dict[str, float] = {
        "abstract": 2.0,
        "executive summary": 2.0,
        "introduction": 1.5,
        "methodology": 3.5,
        "methods": 3.0,
        "data collection": 3.5,
        "sampling": 3.5,
        "weight": 3.0,
        "limitations": 3.0,
        "bias": 3.0,
        "quality": 2.0,
        "variables": 2.0,
        "codebook": 2.0,
        "license": 2.5,
        "access": 2.0,
        "update": 2.0,
        "conclusion": 1.5,
    }

    def __init__(
        self,
        client: Any,
        model_name: str = "gpt-4o-mini",
        *,
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
        max_chunks: int = 12,
    ) -> None:
        self.client = client
        self.model = model_name
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks

        prompts = load_prompts().get("documentation_profiler", {})
        self._system_message = prompts.get(
            "system_message",
            "You extract and summarize supplemental dataset documentation into a concise profile.",
        ).strip()
        self._summary_prompt = prompts.get(
            "summary_prompt",
            (
                "You will receive documentation excerpts for a dataset.\n"
                "{profile_template}\n"
                "Use only the evidence provided. If a field is unknown, return an empty string for it.\n"
                "Documentation excerpts:\n{documentation_chunks}\n\n"
                "Respond with a JSON object following the template. Do not add commentary."
            ),
        )
        self._profile_template = prompts.get(
            "profile_template",
            (
                "Fill this JSON schema with short phrases (<=3 sentences per field):\n"
                "{\n"
                '  "purpose_scope": "",\n'
                '  "provenance_source": "",\n'
                '  "data_collection_method": "",\n'
                '  "population_scope": "",\n'
                '  "variables_overview": "",\n'
                '  "methodology": "",\n'
                '  "quality_limits_bias": "",\n'
                '  "update_frequency": "",\n'
                '  "licensing_access": "",\n'
                '  "usage_guidance": "",\n'
                '  "known_issues": "",\n'
                '  "contact_citation": "",\n'
                '  "source_refs": ""\n'
                "}"
            ),
        )

    # --------------------------
    # Public API
    # --------------------------
    def build_profile_from_path(self, path: str) -> str:
        """Extract, rank, and summarize documentation from a file."""

        text_by_page = self._extract_text(path)
        if not text_by_page:
            return self._empty_profile()

        chunks = self._chunk_document(text_by_page)
        ranked_chunks = self._rank_chunks(chunks)
        return self._summarize_chunks(ranked_chunks)

    def build_profile_from_text(self, text: str) -> str:
        """Build a documentation profile directly from raw text."""

        if not text.strip():
            return self._empty_profile()
        text_by_page = [(None, text)]
        chunks = self._chunk_document(text_by_page)
        ranked_chunks = self._rank_chunks(chunks)
        return self._summarize_chunks(ranked_chunks)

    # --------------------------
    # Extraction helpers
    # --------------------------
    def _extract_text(self, path: str) -> List[tuple[int | None, str]]:
        path_lower = path.lower()
        if path_lower.endswith(".pdf"):
            return self._extract_text_from_pdf(path)
        if path_lower.endswith(".txt") or path_lower.endswith(".md"):
            return self._extract_text_from_text_file(path)
        raise ValueError(f"Unsupported documentation format for {path}. Use PDF, TXT, or MD.")

    def _extract_text_from_pdf(self, path: str) -> List[tuple[int, str]]:
        reader = PdfReader(path)
        pages: List[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            pages.append((idx + 1, page_text))
        return pages

    def _extract_text_from_text_file(self, path: str) -> List[tuple[None, str]]:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        normalized = self._strip_markdown(text)
        return [(None, normalized)]

    # --------------------------
    # Chunking helpers
    # --------------------------
    def _chunk_document(self, text_by_page: Iterable[tuple[int | None, str]]) -> List[DocumentationChunk]:
        chunks: List[DocumentationChunk] = []
        for page, text in text_by_page:
            if not text.strip():
                continue
            page_chunks = self._chunk_text(text, page)
            chunks.extend(page_chunks)
        return chunks

    def _chunk_text(self, text: str, page: int | None) -> List[DocumentationChunk]:
        heading_pattern = re.compile(r"^(#+\s+|\d+\.\s+|[A-Z][A-Za-z\s]{3,})$")
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks: List[DocumentationChunk] = []

        buffer: List[str] = []
        current_heading: str | None = None

        def flush_buffer() -> None:
            if not buffer:
                return
            combined = " ".join(buffer)
            chunks.extend(self._split_to_sized_chunks(combined, current_heading, page))
            buffer.clear()

        for line in raw_lines:
            if heading_pattern.match(line):
                flush_buffer()
                current_heading = line
                continue
            buffer.append(line)
        flush_buffer()

        if not chunks:
            chunks.extend(self._split_to_sized_chunks(" ".join(raw_lines), None, page))
        return chunks

    def _split_to_sized_chunks(
        self, text: str, heading: str | None, page: int | None
    ) -> List[DocumentationChunk]:
        words = text.split()
        if not words:
            return []

        max_len = self.chunk_chars
        overlap = self.chunk_overlap
        segments: List[DocumentationChunk] = []
        start = 0

        while start < len(words):
            end = min(len(words), start + max_len)
            segment_words = words[start:end]
            segment_text = " ".join(segment_words)
            segments.append(DocumentationChunk(text=segment_text, page=page, heading=heading))

            if end == len(words):
                break
            start = end - overlap
            if start < 0:
                start = 0
        return segments

    # --------------------------
    # Scoring helpers
    # --------------------------
    def _rank_chunks(self, chunks: List[DocumentationChunk]) -> List[DocumentationChunk]:
        for idx, chunk in enumerate(chunks):
            chunk.score = self._score_chunk(chunk, idx)
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[: self.max_chunks]

    def _score_chunk(self, chunk: DocumentationChunk, index: int) -> float:
        score = 0.0
        heading = (chunk.heading or "").lower()
        text = chunk.text.lower()

        for key, weight in self._HEADING_PRIORS.items():
            if key in heading:
                score += weight

        keyword_hits = sum(text.count(keyword) for keyword in self._KEYWORDS)
        score += 0.5 * keyword_hits

        if index <= 2:
            score += 1.0

        if "abstract" in heading or "summary" in heading:
            score += 1.0

        return score

    # --------------------------
    # Summarization
    # --------------------------
    def _summarize_chunks(self, chunks: List[DocumentationChunk]) -> str:
        if not chunks:
            return self._empty_profile()

        chunk_text = "\n\n".join(
            self._format_chunk_for_prompt(chunk) for chunk in chunks
        )
        prompt = self._summary_prompt.format(
            profile_template=self._profile_template,
            documentation_chunks=chunk_text,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _format_chunk_for_prompt(self, chunk: DocumentationChunk) -> str:
        location = f"p.{chunk.page}" if chunk.page else "p.?"
        heading = chunk.heading if chunk.heading else "unknown heading"
        snippet = chunk.text.strip()
        return f"[{location} | {heading}]\n{snippet}"

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
        return text

    def _empty_profile(self) -> str:
        return (
            "{"
            '"purpose_scope": "", '
            '"provenance_source": "", '
            '"data_collection_method": "", '
            '"population_scope": "", '
            '"variables_overview": "", '
            '"methodology": "", '
            '"quality_limits_bias": "", '
            '"update_frequency": "", '
            '"licensing_access": "", '
            '"usage_guidance": "", '
            '"known_issues": "", '
            '"contact_citation": "", '
            '"source_refs": ""'
            "}"
        )
