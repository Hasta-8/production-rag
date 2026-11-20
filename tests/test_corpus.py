import hashlib

import httpx
import pytest

from rag import corpus
from rag.corpus import Candidate, LicenceClass, Manifest, Paper, classify_licence

RECORD = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
 <GetRecord><record><metadata>
  <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
   <id>{id}</id>
   <version version="v1"><date>Mon, 12 Jun 2017 17:57:34 GMT</date></version>
   <version version="v3"><date>Wed, 02 Aug 2023 00:41:18 GMT</date></version>
   <title>A   Paper
     Title</title>
   <authors>A. Author, B. Author</authors>
   <categories>cs.CL cs.LG</categories>
   <license>{licence}</license>
  </arXivRaw>
 </metadata></record></GetRecord>
</OAI-PMH>"""

ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
 <error code="idDoesNotExist">Not a valid identifier</error>
</OAI-PMH>"""

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
ARXIV = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (CC_BY, LicenceClass.OPEN),
        ("http://creativecommons.org/licenses/by-sa/4.0/", LicenceClass.OPEN),
        ("http://creativecommons.org/publicdomain/zero/1.0/", LicenceClass.OPEN),
        ("http://creativecommons.org/licenses/by-nc-sa/4.0/", LicenceClass.NONCOMMERCIAL),
        ("http://creativecommons.org/licenses/by-nc-nd/4.0/", LicenceClass.NONCOMMERCIAL),
        (ARXIV, LicenceClass.ARXIV_ONLY),
        (None, LicenceClass.ARXIV_ONLY),
    ],
)
def test_classify_licence(url, expected):
    assert classify_licence(url) == expected


def test_parse_record_reads_licence_latest_version_and_year():
    rec = corpus.parse_arxiv_raw(RECORD.format(id="1234.5678", licence=CC_BY))
    assert rec.arxiv_id == "1234.5678"
    assert rec.title == "A Paper Title"
    assert rec.latest_version == "v3"
    assert rec.year == 2017
    assert rec.licence_class == LicenceClass.OPEN


def test_parse_error_response_raises():
    with pytest.raises(LookupError, match="idDoesNotExist"):
        corpus.parse_arxiv_raw(ERROR)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_check_candidates_builds_manifest_and_reports_failures():
    licences = {"1111.1111": CC_BY, "2222.2222": ARXIV}

    def handler(request: httpx.Request) -> httpx.Response:
        arxiv_id = request.url.params["identifier"].split(":")[-1]
        if arxiv_id not in licences:
            return httpx.Response(200, text=ERROR)
        return httpx.Response(200, text=RECORD.format(id=arxiv_id, licence=licences[arxiv_id]))

    cands = [Candidate(arxiv_id=i, topic="t") for i in ["1111.1111", "2222.2222", "9999.9999"]]
    sleeps = []
    manifest, failures = corpus.check_candidates(cands, _client(handler), sleep=sleeps.append)
    assert [p.licence_class for p in manifest.papers] == [
        LicenceClass.OPEN,
        LicenceClass.ARXIV_ONLY,
    ]
    assert [c.arxiv_id for c, _ in failures] == ["9999.9999"]
    assert sleeps == [corpus.REQUEST_DELAY_S] * 2  # polite delay between requests


def _paper(arxiv_id="1111.1111", cls=LicenceClass.OPEN, sha=None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        version="v2",
        title="T",
        authors="A",
        year=2020,
        topic="t",
        licence_url=None,
        licence_class=cls,
        pdf_sha256=sha,
    )


PDF_BYTES = b"%PDF-1.4 fake test content"


def test_download_pins_hash_and_skips_non_redistributable(tmp_path):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=PDF_BYTES)

    manifest = Manifest(papers=[_paper(), _paper("2222.2222", LicenceClass.ARXIV_ONLY)])
    out = corpus.download_papers(manifest, tmp_path, _client(handler), sleep=lambda _s: None)
    assert requested == ["https://arxiv.org/pdf/1111.1111v2"]
    assert out.papers[0].pdf_sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert out.papers[1].pdf_sha256 is None
    assert (tmp_path / "1111.1111v2.pdf").read_bytes() == PDF_BYTES

    # Second run: file present and matching, so nothing is downloaded.
    requested.clear()
    corpus.download_papers(out, tmp_path, _client(handler), sleep=lambda _s: None)
    assert requested == []


def test_download_rejects_changed_file(tmp_path):
    manifest = Manifest(papers=[_paper(sha="0" * 64)])
    handler = lambda _r: httpx.Response(200, content=PDF_BYTES)  # noqa: E731
    with pytest.raises(ValueError, match="does not match the pinned hash"):
        corpus.download_papers(manifest, tmp_path, _client(handler), sleep=lambda _s: None)


def test_download_rejects_non_pdf(tmp_path):
    manifest = Manifest(papers=[_paper()])
    handler = lambda _r: httpx.Response(200, content=b"<html>captcha</html>")  # noqa: E731
    with pytest.raises(ValueError, match="did not return a PDF"):
        corpus.download_papers(manifest, tmp_path, _client(handler), sleep=lambda _s: None)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.yaml"
    corpus.save_manifest(Manifest(papers=[_paper(sha="a" * 64)]), path)
    loaded = corpus.load_manifest(path)
    assert loaded.papers[0].pdf_sha256 == "a" * 64
    assert loaded.papers[0].licence_class == LicenceClass.OPEN
