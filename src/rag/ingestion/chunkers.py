"""Split documents into chunks that each belong to exactly one page."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Bump when chunking logic changes, so the index version changes with it.
CHUNKER_VERSION = "1"

# Paragraph, line, sentence, word, character: the splitter tries them in this order.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    chunk_index: int
    token_count: int
    content_hash: str


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")  # tokenizer of text-embedding-3-*


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def chunk_id_for(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index:05d}"


def chunk_pages(
    doc_id: str,
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    """Recursive character splitting, page by page.

    Splitting each page on its own means a chunk never spans two pages, so every
    citation points at one exact page. The cost is that a sentence crossing a page
    break is cut; overlap within the page limits the damage.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
        strip_whitespace=True,
    )
    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            if not piece.strip():
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(doc_id, index),
                    doc_id=doc_id,
                    text=piece,
                    page=page,
                    chunk_index=index,
                    token_count=token_counter(piece),
                    content_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                )
            )
    return chunks
