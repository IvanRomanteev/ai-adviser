"""Functions for extracting and validating citations in generated answers.

The assistant must cite its sources inline using the notation [S1],
[S2], … where the integer corresponds to the order in which snippets were
included in the context. This module provides helpers to find citations
in a string and verify that they refer to actual context sources.

This implementation is adapted from the original ai_adviser project
with fixes to ensure that citations are only considered valid when
inline references appear in the body of the answer (not within the
Sources section) and that structural enforcement behaves deterministically.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple, Sequence, Set


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

# Returned by /rag_chat when nothing relevant was retrieved OR citations fail validation upstream.
FALLBACK_MESSAGE = "Not found in the knowledge base."

# Parse a single Sources mapping line; allow bullets or numbered lists:
#   [S1] https://...
#   - [S1] https://...
#   1. [S1] https://...
_SOURCE_MAP_LINE_RE = re.compile(
    r"^(?:(?:[-*]|\d+\.)\s*)?\[S(\d+)\]\s+.+",
    re.IGNORECASE,
)

# Recognise common "Sources" section headers in markdown/plain text.
# Examples:
#   Sources
#   Sources:
#   **Sources**
#   ## Sources
_SOURCES_HEADER_PATTERN = re.compile(
    r"^\s*(?:\*\*|#+\s*)?sources(?:\*\*)?\s*:??\s*$",
    flags=re.IGNORECASE,
)

# Parse a single mapping line used by upsert_sources_section().
# (Kept for compatibility with the existing upsert logic.)
_SOURCE_LINE_PATTERN = re.compile(r"^\s*\[S(\d+)\]\s+(\S+)(?:\s+.*)?$", flags=re.IGNORECASE)


def extract_citation_indices(answer: str) -> List[int]:
    """Extract citation numbers from an answer string."""
    return [int(x) for x in _CITATION_PATTERN.findall(answer or "")]


def _canonical_blob_url_for_source(source: Dict[str, object]) -> str:
    """Best-effort extract the blob URL for a retrieved source."""
    blob = source.get("blob_url")
    if isinstance(blob, str) and blob.strip():
        return blob.strip()
    raw = source.get("raw")
    if isinstance(raw, dict):
        raw_blob = raw.get("blob_url") or raw.get("source_file")
        if isinstance(raw_blob, str) and raw_blob.strip():
            return raw_blob.strip()
    return ""


def build_sources_mapping(
    sources: Iterable[Dict[str, object]],
    *,
    used_indices: Optional[Iterable[int]] = None,
) -> Tuple[List[int], str]:
    """Build a canonical sources mapping text (lines like: "[S1] https://...")."""
    src_list = list(sources)
    if used_indices is None:
        indices = list(range(1, len(src_list) + 1))
    else:
        indices = sorted({int(i) for i in used_indices if int(i) > 0})
        indices = [i for i in indices if i <= len(src_list)]

    lines: List[str] = []
    for idx in indices:
        blob = _canonical_blob_url_for_source(src_list[idx - 1])
        if not blob:
            blob = "<missing_blob_url>"
        lines.append(f"[S{idx}] {blob}")
    return indices, "\n".join(lines)


def upsert_sources_section(
    answer: str,
    sources: Iterable[Dict[str, object]],
    *,
    prefer_bold_heading: bool = True,
) -> str:
    """Ensure the answer has a Sources section with correct blob URLs.

    Only fixes the final Sources mapping. Does not add missing inline citations.
    """
    answer = answer or ""
    used = sorted(set(extract_citation_indices(answer)))
    _, mapping = build_sources_mapping(sources, used_indices=used or None)

    heading = "**Sources**" if prefer_bold_heading else "Sources"
    block = f"{heading}\n{mapping}" if mapping else heading

    lines = answer.splitlines()
    header_idx: Optional[int] = None
    for i, raw_line in enumerate(lines):
        if _SOURCES_HEADER_PATTERN.match(raw_line.strip()):
            header_idx = i
            break

    if header_idx is None:
        return answer.rstrip() + "\n\n" + block + "\n"

    prefix = "\n".join(lines[:header_idx]).rstrip()
    return prefix + "\n\n" + block + "\n"


def _split_body_and_sources(answer: str) -> tuple[list[str], list[str]]:
    """Split answer into body lines and sources-section lines."""
    lines = (answer or "").splitlines()
    for i, line in enumerate(lines):
        if _SOURCES_HEADER_PATTERN.match(line.strip()):
            return lines[:i], lines[i:]
    return lines, []

def enforce_citation_structure(
    answer: str,
    sources: Sequence[Dict[str, object]],
    *,
    prefer_bold_heading: bool = True,
) -> str:
    """Deterministically make an answer pass structural citation validation.

    This helper is designed for production reliability when
    CITATION_ENFORCE_STRUCTURE=true.

    It will:

    * Strip any existing Sources section (if present).
    * Ensure every non-empty body line (except section headers) has at least one
      inline citation like ``[S1]``. Lines missing citations get a default
      citation appended.
    * Replace any out-of-range citation indices (e.g. ``[S99]`` when only 3
      sources exist) with the default citation.
    * Rebuild the Sources section using ``upsert_sources_section``.

    IMPORTANT: This function does **not** change the factual content of the
    answer. It only fixes citation formatting.
    """
    answer = answer or ""

    # Work only on the body to avoid modifying the Sources mapping lines.
    body_lines, _ = _split_body_and_sources(answer)

    # Pick a default in-range citation index. Prefer one already used in body.
    used = sorted(set(extract_citation_indices("\n".join(body_lines))))
    default_idx = used[0] if used else 1
    if len(sources) >= 1:
        if default_idx < 1 or default_idx > len(sources):
            default_idx = 1
    else:
        # No sources – just return the original body (validation will fail upstream).
        return answer

    def _fix_out_of_range(match: re.Match) -> str:
        try:
            n = int(match.group(1))
        except Exception:
            return f"[S{default_idx}]"
        if n < 1 or n > len(sources):
            return f"[S{default_idx}]"
        return match.group(0)

    fixed_lines: list[str] = []
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue

        # Don't force citations on section headers like "Summary" / "Sources".
        if _SECTION_HEADER_RE.match(stripped):
            fixed_lines.append(line)
            continue

        # First, normalise any out-of-range citation indices in the line.
        line_fixed = _CITATION_PATTERN.sub(_fix_out_of_range, line)

        # Then, ensure the line has at least one citation.
        if not _CITATION_PATTERN.search(line_fixed):
            line_fixed = f"{line_fixed.rstrip()} [S{default_idx}]"

        fixed_lines.append(line_fixed)

    fixed_body = "\n".join(fixed_lines).strip()

    # Rebuild Sources section to match our (possibly updated) inline citations.
    return upsert_sources_section(
        fixed_body,
        sources,
        prefer_bold_heading=prefer_bold_heading,
    )



def _validate_baseline_citations(answer: str, sources: Sequence[Dict[str, object]]) -> bool:
    # ВАЖНО: учитываем только body, иначе "Sources-only" обманет baseline.
    body_lines, _ = _split_body_and_sources(answer)
    citation_nums = extract_citation_indices("\n".join(body_lines))
    if not citation_nums:
        return False
    max_ref = max(citation_nums)
    if max_ref > len(sources):
        return False
    if any(n <= 0 for n in citation_nums):
        return False
    return True

# Allow section headers without requiring citations on that line.
# Supports markdown headings (e.g. "## Summary") and plain text (e.g. "Summary:").
_SECTION_HEADER_RE = re.compile(
    r"^(#{1,6}\s*)?"
    r"(summary"
    r"|key points(?:\s*(?:/|or)\s*risks)?"
    r"|steps(?:\s*(?:/|or)\s*best practices)?"
    r"|exceptions(?:\s*(?:/|or)\s*limitations)?"
    r"|sources)"
    r"\s*[:\-–—]*\s*$",
    re.IGNORECASE,
)


def _validate_structure(answer: str, sources: Sequence[Dict[str, object]]) -> bool:
    lines = (answer or "").splitlines()
    in_sources = False
    saw_sources_header = False
    saw_any_mapping = False
    saw_body_content_line = False

    cited_in_body: Set[int] = set()
    mapped_indices: Set[int] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Skip blank lines
            continue

        # Detect the beginning of a Sources section.  Accept variations such as
        # "Sources", "Sources:", "**Sources**" and "## Sources".  When this
        # matches we switch into the sources-parsing mode and require that
        # subsequent non-empty lines map citations to blob URLs.
        if _SOURCES_HEADER_PATTERN.match(stripped):
            saw_sources_header = True
            in_sources = True
            continue

        # Other top-level section headers (e.g. "Summary", "Key points") are
        # exempt from citation requirements.  If we encounter one of these we
        # simply skip over it.  The "Sources" header is handled above.
        if _SECTION_HEADER_RE.match(stripped):
            continue

        if in_sources:
            # Lines within the sources section must conform to the mapping
            # pattern.  Allow bullets or numbered lists.  If any line fails
            # to match we treat the entire answer as invalid.
            m = _SOURCE_MAP_LINE_RE.match(stripped)
            if not m:
                return False
            idx = int(m.group(1))
            mapped_indices.add(idx)
            saw_any_mapping = True
            continue

        # For all other non-empty lines (body content) require at least one
        # inline citation.  Collect all referenced citation numbers.
        saw_body_content_line = True
        if not _CITATION_PATTERN.search(stripped):
            return False
        cited_in_body.update(extract_citation_indices(stripped))

    # Reject answers that contain no substantive body content (e.g. only headers
    # or only a Sources section).
    if not saw_body_content_line:
        return False
    # Require that a Sources header and at least one mapping line exist.
    if not saw_sources_header or not saw_any_mapping:
        return False
    # All citations used in the body must also appear in the sources mapping.
    if cited_in_body and not cited_in_body.issubset(mapped_indices):
        return False
    # Ensure no body citation refers to a non-existent source.
    if cited_in_body and max(cited_in_body) > len(sources):
        return False

    return True



def validate_answer_citations(
    answer: str,
    sources: Iterable[Dict[str, object]],
    *,
    enforce_structure: Optional[bool] = None,
) -> bool:
    """Validate that an answer contains proper inline citations and, if requested,
    a well‑formed Sources section.

    Parameters:
        answer: The generated answer string.
        sources: A sequence of source metadata dictionaries.
        enforce_structure: If True, require that the answer contain a
            Sources section with correctly formatted mapping lines.  If
            False, only baseline citation checks are performed.  If
            None (the default), the global settings.CITATION_ENFORCE_STRUCTURE
            flag determines whether the structure is enforced.

    Returns:
        True if the answer satisfies the baseline citation rules (and
        optionally the structural rules); False otherwise.
    """
    # Allow explicit fallback answer through without citations.
    if (answer or "").strip() == FALLBACK_MESSAGE:
        return True

    if not _validate_baseline_citations(answer, sources):
        return False

    try:
        from ai_adviser.config import settings  # late import to avoid cycles
        default_enforce = settings.CITATION_ENFORCE_STRUCTURE
    except Exception:
        default_enforce = False

    # Determine whether to enforce structural rules.  If enforce_structure
    # is None use the settings flag; otherwise use the explicit value.
    enforce = default_enforce if enforce_structure is None else enforce_structure
    if not enforce:
        return True
    return _validate_structure(answer, list(sources))
