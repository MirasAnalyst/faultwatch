"""Machinery-casualty corpus: public marine accident investigation reports.

Sources (all free to reuse):
  * NTSB marine investigation reports and accident briefs (US government
    work, public domain). Each carries a structured 'Accident/Casualty type'
    field, used as ground truth for the triage eval.
  * UK MAIB investigation reports (Crown copyright, Open Government Licence
    v3.0 - attribution kept in the manifest).
  * NSIA (Norway) report on the Viking Sky blackout, 2019 - the closest public
    case to a cruise-ship loss of propulsion.

Only machinery-related casualties are kept: engine-room fires, loss of
propulsion or power, blackouts, steering failures, machinery damage.

    python scripts/download_data.py incidents     # download + build corpus/
"""
from __future__ import annotations

import gzip
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
NTSB_URL = "https://www.ntsb.gov/investigations/AccidentReports/Reports/{}.pdf"
MAIB_SEARCH = ("https://www.gov.uk/api/search.json?filter_format=maib_report&count=1000&start={}"
               "&fields=title,link,public_timestamp,vessel_type,report_type")
GOVUK_CONTENT = "https://www.gov.uk/api/content{}"
NSIA_VIKING_SKY = "https://nsia.no/Marine/Published-reports/2024-05?pid=SHT-Report-ReportFile&attach=1"
LICENSES = {"ntsb": "Public domain (US government work)",
            "maib": "Contains public sector information licensed under the Open Government Licence v3.0 (MAIB)",
            "nsia": "NSIA report, reproduced for research with citation"}

MACHINERY_TITLE = re.compile(r"fire|explosion|engine|propulsion|blackout|\bpower\b|steering|machinery|generator|"
                             r"turbo|boiler|crankcase|thruster|electrical|fuel", re.I)
STRONG_TERMS = re.compile(r"loss of propulsion|lost propulsion|blackout|loss of (?:electrical )?power|"
                          r"loss of steering|steering (?:gear )?failure|engine room fire|main engine failure", re.I)
MACHINERY_TYPES = re.compile(r"fire|explosion|machinery|propulsion|equipment|power", re.I)

SECTION_HEADINGS = [
    "Background", "Event Sequence", "Accident Events", "Additional Information", "Postaccident Actions",
    "Analysis", "Probable Cause", "Lessons Learned", "Conclusions", "Findings", "Recommendations",
    "Synopsis", "Summary", "Factual Information", "Narrative", "Safety Lessons", "Actions Taken",
    "Safety Issues", "Analysis and Findings", "Contributing", "Engine Room", "Fire", "Steering",
]
HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(" + "|".join(SECTION_HEADINGS) + r")\b[\w ,/-]{0,40}$", re.I)


def _get(url: str, timeout: int = 60) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.headers.get("Content-Type", "")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# download
def download_ntsb(raw: Path, years=range(16, 27), max_number=60, misses_to_stop=10, pause=0.25):
    """NTSB marine reports are named MIR-yy/nn (investigation reports) and
    MAB-yy/nn (accident briefs); probe each series until it runs out."""
    out = raw / "ntsb"
    out.mkdir(parents=True, exist_ok=True)
    for yy in years:
        for kind in ("MIR", "MAB"):
            misses = 0
            for nn in range(1, max_number + 1):
                rid = f"{kind}{yy:02d}{nn:02d}"
                f = out / f"{rid}.pdf"
                if f.exists():
                    misses = 0
                    continue
                try:
                    body, ctype = _get(NTSB_URL.format(rid))
                except Exception:
                    body, ctype = b"", ""
                if "pdf" in ctype and body[:4] == b"%PDF":
                    f.write_bytes(body)
                    misses = 0
                    print(f"   ntsb {rid}")
                else:
                    misses += 1
                    if misses >= misses_to_stop:
                        break
                time.sleep(pause)


def download_maib(raw: Path, pause=0.25):
    out = raw / "maib"
    out.mkdir(parents=True, exist_ok=True)
    results, start = [], 0
    while True:
        page = json.loads(_get(MAIB_SEARCH.format(start))[0])["results"]
        results += page
        if len(page) < 1000:
            break
        start += 1000
    meta = []
    for r in results:
        if not MACHINERY_TITLE.search(r["title"]):
            continue
        slug = r["link"].rstrip("/").split("/")[-1][:120]
        f = out / f"{slug}.pdf"
        if not f.exists():
            try:
                content = json.loads(_get(GOVUK_CONTENT.format(r["link"]))[0])
                atts = content.get("details", {}).get("attachments", [])
                pdfs = [a for a in atts if str(a.get("url", "")).lower().endswith(".pdf")]
                if not pdfs:
                    continue
                pdfs.sort(key=lambda a: a.get("file_size", 0) or 0, reverse=True)
                body, _ = _get(pdfs[0]["url"])
                if body[:4] != b"%PDF":
                    continue
                f.write_bytes(body)
                print(f"   maib {slug[:70]}")
                time.sleep(pause)
            except Exception as e:                   # one bad report must not stop the run
                print(f"   maib skip {slug[:50]}: {e}")
                continue
        meta.append({"file": f.name, "title": r["title"], "date": r.get("public_timestamp", "")[:10],
                     "vessel_type": ";".join(r.get("vessel_type") or []),
                     "url": "https://www.gov.uk" + r["link"]})
    pd.DataFrame(meta).to_csv(out / "_index.csv", index=False)


def download_nsia(raw: Path):
    out = raw / "nsia"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "viking_sky_2024-05.pdf"
    if not f.exists():
        body, _ = _get(NSIA_VIKING_SKY, timeout=180)
        if body[:4] == b"%PDF":
            f.write_bytes(body)


# ---------------------------------------------------------------------------
# text and metadata
def pdf_title(path: Path) -> str:
    from pypdf import PdfReader
    try:
        md = PdfReader(str(path)).metadata
        return (md.title or "").strip() if md else ""
    except Exception:
        return ""


def pdf_text(path: Path, max_pages: int = 120) -> list[str]:
    from pypdf import PdfReader
    pages = []
    for p in PdfReader(str(path)).pages[:max_pages]:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        t = (t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
              .replace("’", "'").replace("–", "-").replace("—", "-"))
        pages.append(t)
    return pages


def _ntsb_meta(rid: str, pages: list[str], meta_title: str = "") -> dict:
    text = "\n".join(pages[:3])
    m = re.search(r"(?:Accident|Casualty)\s+[Tt]ype\s+([A-Za-z/ ,\-]+?)(?=\s+No\.|\s*\n|\s{2,})", text)
    date = re.search(r"On ([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    title = ""
    if len(pages) > 1:
        first = next((l for l in pages[1].splitlines() if l.strip()), "")
        title = re.sub(r"\s*(NTSB/)?(MIR|MAB)[- ]?\d{2}\s*[/-]\s*\d{2}.*$", "", first).strip()
        title = re.sub(r"^\s*\d+\s+", "", title)
    if meta_title and len(meta_title) > 10:          # the PDF's own title is the most reliable
        title = meta_title
    if not title:
        title = next((l.strip() for l in pages[0].splitlines() if len(l.strip()) > 20), rid)
    return {"doc_id": f"NTSB-{rid}", "source": "ntsb", "title": title[:200],
            "casualty_type": m.group(1).strip() if m else "",
            "date": pd.to_datetime(date.group(1)).date().isoformat() if date else "",
            "url": NTSB_URL.format(rid), "license": LICENSES["ntsb"]}


def is_machinery(meta: dict, text: str) -> bool:
    if MACHINERY_TYPES.search(meta.get("casualty_type", "")):
        return True
    strong = len(STRONG_TERMS.findall(text))
    return strong >= 3 or (bool(MACHINERY_TITLE.search(meta.get("title", ""))) and strong >= 1)


RUNNING_HEADER = re.compile(r"(?:NTSB/)?(?:MIR|MAB)[- ]?\d{2}\s*[-/ ]\s*\d{2}|MARINE ACCIDENT INVESTIGATION BRANCH|"
                            r"^\s*(?:Page \d+|\d+ of \d+)\s*$", re.I)


def chunk_document(doc_id: str, pages: list[str], size: int = 1400, overlap: int = 250,
                   title: str = "") -> list[dict]:
    """Section-aware chunks: paragraphs are packed up to `size` characters,
    each chunk is tagged with the last section heading seen. Running page
    headers (report number, title repeated on every page) are dropped."""
    section, buf, chunks = "Summary", "", []
    short_title = title[:40].lower()

    def flush():
        nonlocal buf
        t = re.sub(r"[ \t]+", " ", buf).strip()
        if len(t) > 200:
            chunks.append({"doc_id": doc_id, "section": section, "text": t})
        buf = t[-overlap:] if len(t) > overlap else ""

    for page in pages:
        for line in page.splitlines():
            s = line.strip()
            if not s or re.fullmatch(r"\d{1,3}", s):
                continue
            if len(s) < 140 and (RUNNING_HEADER.search(s) or (short_title and s.lower().startswith(short_title))):
                continue
            if title:                       # header glued into a body line by the PDF extractor
                s = re.sub(re.escape(title) + r"\s*(?:NTSB/)?(?:MIR|MAB)[- ]?\d{2}\s*[-/ ]\s*\d{2}", " ", s)
            s = re.sub(r"\s(?:NTSB/)?(?:MIR|MAB)-\d{2}[-/]\d{2}\s", " ", s)
            h = HEADING_RE.match(s)
            if h and len(s) < 60:
                flush()
                buf = ""
                section = h.group(1).title()
                continue
            buf += (" " if buf.endswith(("-", " ")) is False else "") + s
            if len(buf) >= size:
                flush()
    flush()
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{doc_id}#{i:03d}"
    return chunks


def build(raw: Path, out: Path) -> pd.DataFrame:
    """Raw PDFs -> corpus/manifest.csv + corpus/chunks.jsonl.gz (machinery reports only)."""
    out.mkdir(parents=True, exist_ok=True)
    docs, chunks = [], []
    for f in sorted((raw / "ntsb").glob("*.pdf")):
        pages = pdf_text(f)
        meta = _ntsb_meta(f.stem, pages, pdf_title(f))
        text = "\n".join(pages)
        if is_machinery(meta, text):
            docs.append({**meta, "pages": len(pages), "chars": len(text)})
            chunks += chunk_document(meta["doc_id"], pages, title=meta["title"])
    idx = raw / "maib" / "_index.csv"
    if idx.exists():
        for r in pd.read_csv(idx).fillna("").itertuples():
            f = raw / "maib" / r.file
            if not f.exists():
                continue
            pages = pdf_text(f)
            text = "\n".join(pages)
            meta = {"doc_id": "MAIB-" + Path(r.file).stem[:60], "source": "maib", "title": r.title,
                    "casualty_type": "", "date": r.date, "url": r.url, "license": LICENSES["maib"],
                    "vessel_type": r.vessel_type}
            if is_machinery(meta, text):
                docs.append({**meta, "pages": len(pages), "chars": len(text)})
                chunks += chunk_document(meta["doc_id"], pages, title=meta["title"])
    f = raw / "nsia" / "viking_sky_2024-05.pdf"
    if f.exists():
        pages = pdf_text(f)
        meta = {"doc_id": "NSIA-2024-05-VikingSky", "source": "nsia",
                "title": "Loss of propulsion and near-grounding of the cruise ship Viking Sky, Hustadvika, 2019",
                "casualty_type": "Loss of propulsion", "date": "2019-03-23", "url": NSIA_VIKING_SKY,
                "license": LICENSES["nsia"]}
        docs.append({**meta, "pages": len(pages), "chars": sum(map(len, pages))})
        chunks += chunk_document(meta["doc_id"], pages, title=meta["title"])
    manifest = pd.DataFrame(docs)
    manifest.to_csv(out / "manifest.csv", index=False)
    with gzip.open(out / "chunks.jsonl.gz", "wt", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    return manifest


def load_chunks(corpus_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(corpus_dir)
    with gzip.open(d / "chunks.jsonl.gz", "rt", encoding="utf-8") as fh:
        chunks = pd.DataFrame([json.loads(l) for l in fh])
    return chunks, pd.read_csv(d / "manifest.csv").fillna("")
