"""Corpus management: licence checks against arXiv metadata, the manifest, and downloads.

Only papers whose licence allows redistribution are served by the demo. The licence
of each paper is read from arXiv's OAI-PMH metadata (``arXivRaw`` format), never
assumed.
"""

from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, Field

OAI_URL = "https://oaipmh.arxiv.org/oai"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}{version}"
ABS_URL = "https://arxiv.org/abs/{arxiv_id}{version}"
USER_AGENT = "productionrag-corpus/0.1 (portfolio project; polite single-threaded client)"
REQUEST_DELAY_S = 3.0  # arXiv asks automated clients to wait about 3 seconds between requests

_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}


class LicenceClass(StrEnum):
    """How freely a paper can be redistributed by this project."""

    OPEN = "open"  # CC BY, CC BY-SA, CC0: redistribution allowed with attribution
    NONCOMMERCIAL = "noncommercial"  # CC BY-NC-*: redistribution allowed, non-commercial only
    ARXIV_ONLY = "arxiv_only"  # arXiv's default licence: only arXiv may distribute it


def classify_licence(url: str | None) -> LicenceClass:
    if not url:
        return LicenceClass.ARXIV_ONLY
    u = url.lower()
    if "creativecommons.org" in u:
        if "publicdomain/zero" in u:
            return LicenceClass.OPEN
        if "/by-nc" in u:
            return LicenceClass.NONCOMMERCIAL
        if "/by/" in u or "/by-sa/" in u:
            return LicenceClass.OPEN
    return LicenceClass.ARXIV_ONLY


@dataclass(frozen=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    authors: str
    year: int
    latest_version: str
    licence_url: str | None
    categories: str

    @property
    def licence_class(self) -> LicenceClass:
        return classify_licence(self.licence_url)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_arxiv_raw(xml_text: str) -> ArxivRecord:
    """Parse one OAI-PMH ``GetRecord`` response in ``arXivRaw`` format."""
    root = ET.fromstring(xml_text)
    error = root.find("oai:error", _NS)
    if error is not None:
        raise LookupError(f"arXiv OAI error {error.get('code')}: {_clean(error.text)}")
    raw = root.find(".//raw:arXivRaw", _NS)
    if raw is None:
        raise LookupError("No arXivRaw record in the response")
    versions = raw.findall("raw:version", _NS)
    if not versions:
        raise LookupError("Record has no versions")
    first_date = parsedate_to_datetime(_clean(versions[0].findtext("raw:date", "", _NS)))
    return ArxivRecord(
        arxiv_id=_clean(raw.findtext("raw:id", "", _NS)),
        title=_clean(raw.findtext("raw:title", "", _NS)),
        authors=_clean(raw.findtext("raw:authors", "", _NS)),
        year=first_date.year,
        latest_version=versions[-1].get("version", "v1"),
        licence_url=_clean(raw.findtext("raw:license", "", _NS)) or None,
        categories=_clean(raw.findtext("raw:categories", "", _NS)),
    )


def fetch_record(client: httpx.Client, arxiv_id: str) -> ArxivRecord:
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:arXiv.org:{arxiv_id}",
        "metadataPrefix": "arXivRaw",
    }
    response = client.get(OAI_URL, params=params)
    response.raise_for_status()
    return parse_arxiv_raw(response.text)


# ---------------------------------------------------------------- manifest


class Candidate(BaseModel):
    arxiv_id: str
    topic: str


class Paper(BaseModel):
    arxiv_id: str
    version: str
    title: str
    authors: str
    year: int
    topic: str
    licence_url: str | None
    licence_class: LicenceClass
    pdf_sha256: str | None = None

    @property
    def pdf_url(self) -> str:
        return PDF_URL.format(arxiv_id=self.arxiv_id, version=self.version)

    @property
    def abs_url(self) -> str:
        return ABS_URL.format(arxiv_id=self.arxiv_id, version=self.version)

    @property
    def filename(self) -> str:
        return f"{self.arxiv_id}{self.version}.pdf"

    @property
    def can_redistribute(self) -> bool:
        return self.licence_class in (LicenceClass.OPEN, LicenceClass.NONCOMMERCIAL)


class Manifest(BaseModel):
    papers: list[Paper] = Field(default_factory=list)


def load_candidates(path: Path) -> list[Candidate]:
    data = yaml.safe_load(path.read_text()) or {}
    return [Candidate(**c) for c in data.get("candidates", [])]


def load_manifest(path: Path) -> Manifest:
    if not path.exists():
        return Manifest()
    return Manifest(**(yaml.safe_load(path.read_text()) or {}))


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by `rag corpus check` from arXiv metadata. Do not edit licence fields by\n"
        "# hand. pdf_sha256 is filled in by `rag corpus download` and pins the exact file.\n"
    )
    body = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    path.write_text(header + body)


def check_candidates(
    candidates: Iterable[Candidate],
    client: httpx.Client,
    existing: Manifest | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str], None] = lambda _msg: None,
) -> tuple[Manifest, list[tuple[Candidate, str]]]:
    """Look up every candidate on arXiv. Returns the manifest and a list of failures."""
    known = {p.arxiv_id: p for p in (existing.papers if existing else [])}
    papers: list[Paper] = []
    failures: list[tuple[Candidate, str]] = []
    for i, cand in enumerate(candidates):
        if i:
            sleep(REQUEST_DELAY_S)
        try:
            rec = fetch_record(client, cand.arxiv_id)
        except (httpx.HTTPError, LookupError, ET.ParseError) as exc:
            failures.append((cand, str(exc)))
            on_progress(f"FAIL  {cand.arxiv_id}: {exc}")
            continue
        old = known.get(rec.arxiv_id)
        same_file = old is not None and old.version == rec.latest_version
        papers.append(
            Paper(
                arxiv_id=rec.arxiv_id,
                version=rec.latest_version,
                title=rec.title,
                authors=rec.authors,
                year=rec.year,
                topic=cand.topic,
                licence_url=rec.licence_url,
                licence_class=rec.licence_class,
                pdf_sha256=old.pdf_sha256 if same_file else None,
            )
        )
        on_progress(f"{rec.licence_class.value:<14} {rec.arxiv_id}  {rec.title[:70]}")
    return Manifest(papers=papers), failures


# ---------------------------------------------------------------- downloads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_papers(
    manifest: Manifest,
    pdf_dir: Path,
    client: httpx.Client,
    include: Callable[[Paper], bool] = lambda p: p.can_redistribute,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str], None] = lambda _msg: None,
) -> Manifest:
    """Download selected papers, pinning each file by SHA-256 in the returned manifest.

    A file that already exists and matches its recorded hash is not downloaded again.
    A mismatch raises, because the pinned file has changed.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    updated: list[Paper] = []
    fetched = 0
    for paper in manifest.papers:
        if not include(paper):
            updated.append(paper)
            continue
        target = pdf_dir / paper.filename
        if target.exists() and paper.pdf_sha256 and sha256_file(target) == paper.pdf_sha256:
            updated.append(paper)
            on_progress(f"ok     {paper.filename}")
            continue
        if fetched:
            sleep(REQUEST_DELAY_S)
        response = client.get(paper.pdf_url, follow_redirects=True)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"{paper.pdf_url} did not return a PDF")
        digest = hashlib.sha256(response.content).hexdigest()
        if paper.pdf_sha256 and digest != paper.pdf_sha256:
            raise ValueError(
                f"{paper.filename}: downloaded file hash {digest} does not match the pinned "
                f"hash {paper.pdf_sha256}"
            )
        tmp = target.with_suffix(".part")
        tmp.write_bytes(response.content)
        tmp.replace(target)
        fetched += 1
        updated.append(paper.model_copy(update={"pdf_sha256": digest}))
        on_progress(f"saved  {paper.filename} ({len(response.content) // 1024} KB)")
    return Manifest(papers=updated)


def http_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0)
