import math
import re
import statistics

import tiktoken

from app.config import get_settings
from app.services.embedding_service import generate_embeddings

settings = get_settings()

_ENCODING = tiktoken.get_encoding("cl100k_base")

# A document is considered "structured" when it contains at least this many
# recognizable section headings (numbered "5.1" style or DOCX Heading styles).
MIN_HEADINGS_FOR_SECTIONS = 2

# Below this similarity we treat two semantic segments as a topic boundary.
SEMANTIC_BREAK_THRESHOLD = 0.5


# ── Heading detection ────────────────────────────────────────────────────────

_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.\)]?\s+(.+)$")
_HEADING_STYLE_RE = re.compile(r"^\[(Heading\s*\d+)\]\s*(.*)$", re.IGNORECASE)


def _parse_heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) if a line looks like a section heading, else None."""
    s = line.strip()
    if not s:
        return None

    m = _HEADING_STYLE_RE.match(s)
    if m:
        title = m.group(2).strip()
        if not title:
            return None
        level = int(re.search(r"\d+", m.group(1)).group())
        return level, title

    m = _NUMBERED_HEADING_RE.match(s)
    if m:
        num, title = m.group(1), m.group(2).strip()
        if title and len(title) <= 120:
            level = num.count(".") + 1
            return level, title

    return None


def _count_headings(text: str) -> int:
    return sum(1 for line in text.split("\n") if _parse_heading(line))


# ── Token helpers ────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Split text into hard token-bounded pieces (last resort, no boundaries)."""
    if _token_count(text) <= max_tokens:
        return [text]
    tokens = _ENCODING.encode(text)
    out: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        out.append(_ENCODING.decode(tokens[start:end]).strip())
        start = end
    return out


def _split_into_pieces(text: str, max_tokens: int) -> list[str]:
    """Greedily pack sentences into pieces no larger than max_tokens."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    pieces: list[str] = []
    current = ""
    for s in sentences:
        if not current:
            current = s
        elif _token_count(current + " " + s) <= max_tokens:
            current += " " + s
        else:
            pieces.append(current)
            current = s
    if current:
        pieces.append(current)

    result: list[str] = []
    for p in pieces:
        result.extend(_hard_split(p, max_tokens))
    return result


# ── Section-aware chunking (structured documents) ────────────────────────────

def _chunk_by_sections(text: str, page_number: int) -> list[dict]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []  # (heading_path, body)
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if any(l.strip() for l in body):
            path = " > ".join(t for _, t in stack)
            sections.append((path, "\n".join(body)))
        body = []

    for line in text.split("\n"):
        h = _parse_heading(line)
        if h:
            flush()
            level, title = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        elif line.strip():
            body.append(line)

    flush()

    chunks: list[dict] = []
    for path, body_text in sections:
        content = f"{path}\n{body_text}" if path else body_text
        if _token_count(content) <= settings.CHUNK_SIZE:
            chunks.append({"content": content.strip(), "page_number": page_number})
        else:
            for piece in _split_into_pieces(body_text, settings.CHUNK_SIZE):
                c = f"{path}\n{piece}" if path else piece
                chunks.append({"content": c.strip(), "page_number": page_number})
    return chunks


# ── Semantic chunking (messy / unstructured documents) ───────────────────────

def _segment_text(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    segments: list[str] = []
    for p in paras:
        if _token_count(p) > 150:
            segments.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())
        else:
            segments.append(p)
    return segments


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _find_breakpoints(sims: list[float]) -> list[int]:
    """Return segment indices after which a topic boundary should be inserted."""
    if not sims:
        return []
    mean = statistics.mean(sims)
    std = statistics.pstdev(sims) if len(sims) > 1 else 0.0
    cutoff = max(0.35, mean - 0.5 * std)
    bps: list[int] = []
    for i, s in enumerate(sims):
        left = sims[i - 1] if i > 0 else 1.0
        right = sims[i + 1] if i + 1 < len(sims) else 1.0
        if s < cutoff or (s < left and s < right and s < SEMANTIC_BREAK_THRESHOLD):
            bps.append(i)
    return bps


async def _chunk_semantic(text: str, page_number: int) -> list[dict]:
    segments = _segment_text(text)
    if len(segments) <= 1:
        return [
            {"content": p.strip(), "page_number": page_number}
            for p in _split_into_pieces(text, settings.CHUNK_SIZE)
        ]

    embeddings = await generate_embeddings(segments)
    sims = [_cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    breakpoints = _find_breakpoints(sims)

    groups: list[str] = []
    start = 0
    for bp in breakpoints:
        groups.append(" ".join(segments[start : bp + 1]).strip())
        start = bp + 1
    groups.append(" ".join(segments[start:]).strip())

    chunks: list[dict] = []
    for g in groups:
        if not g:
            continue
        for piece in _split_into_pieces(g, settings.CHUNK_SIZE):
            chunks.append({"content": piece.strip(), "page_number": page_number})
    return chunks


# ── Fallback: fixed-size window (kept from the previous implementation) ──────

def _chunk_fixed(text: str, page_number: int) -> list[dict]:
    tokens = _ENCODING.encode(text)
    chunks: list[dict] = []
    start = 0
    while start < len(tokens):
        end = min(start + settings.CHUNK_SIZE, len(tokens))
        decoded = _ENCODING.decode(tokens[start:end]).strip()
        if decoded:
            chunks.append({"content": decoded, "page_number": page_number})
        if end >= len(tokens):
            break
        start = end - settings.CHUNK_OVERLAP
    return chunks


# ── Orchestrator ─────────────────────────────────────────────────────────────

async def chunk_pages(pages: list[dict]) -> list[dict]:
    """Chunk extracted pages.

    Chooses the strategy per document (across all pages):
      - Section-aware chunking when the document has section headings.
      - Semantic chunking when it does not (messy / unstructured text).
    """
    total_headings = sum(_count_headings(p["content"]) for p in pages)
    use_sections = total_headings >= MIN_HEADINGS_FOR_SECTIONS

    all_chunks: list[dict] = []
    global_index = 0
    for page in pages:
        text = page["content"]
        page_number = page["page_number"]

        if use_sections:
            page_chunks = _chunk_by_sections(text, page_number)
        else:
            try:
                page_chunks = await _chunk_semantic(text, page_number)
            except Exception:
                page_chunks = _chunk_fixed(text, page_number)

        for chunk in page_chunks:
            chunk["chunk_index"] = global_index
            all_chunks.append(chunk)
            global_index += 1

    return all_chunks
