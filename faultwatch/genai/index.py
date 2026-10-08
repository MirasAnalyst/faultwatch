"""Hybrid retrieval over the incident corpus: BM25 (exact engineering terms,
part names, alarm tags) fused with dense embeddings (paraphrase - "lost all
power" vs "blackout") by reciprocal-rank fusion.

Dense back-ends, chosen with `dense=`:
  * "st"    - sentence-transformers all-MiniLM-L6-v2, local (PyTorch)
  * "azure" - Azure OpenAI embeddings deployment (AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
  * "lsa"   - TF-IDF + truncated SVD: no downloads, deterministic (CI default)
  * None    - BM25 only
Embeddings are cached under corpus/.cache so the index builds once.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import load_chunks

STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
with which after before during when while not no but into than then there their they them been being
would could should about over under also only other such these those any all each between""".split())
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-/]*", text.lower()) if t not in STOP and len(t) > 1]


class Retriever:
    def __init__(self, corpus_dir: Path | str = "corpus", dense: str | None = "auto"):
        from rank_bm25 import BM25Okapi
        self.dir = Path(corpus_dir)
        self.chunks, self.manifest = load_chunks(self.dir)
        # the title is prepended so a chunk deep in a report still knows what the report is about
        title = self.chunks.doc_id.map(self.manifest.set_index("doc_id")["title"]).fillna("")
        self._docs = (title + ". " + self.chunks.section + ". " + self.chunks.text).tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in self._docs])
        if dense == "auto":
            dense = "azure" if os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") else "st" if _has_st() else "lsa"
        self.dense_name = dense
        self.E = self._embed_corpus(dense) if dense else None

    # ---- dense ---------------------------------------------------------------
    def _cache(self, name):
        h = hashlib.sha1("".join(self.chunks.chunk_id).encode()).hexdigest()[:10]
        return self.dir / ".cache" / f"emb_{name}_{h}.npy"

    def _embed_corpus(self, dense):
        if dense == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), min_df=2,
                                          sublinear_tf=True, token_pattern=None)
            X = self._tfidf.fit_transform(self._docs)
            self._svd = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=0).fit(X)
            return _norm(self._svd.transform(X))
        f = self._cache(dense)
        if f.exists():
            E = np.load(f)
        else:
            E = self._encode(self._docs, dense)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, E)
        return E

    def _encode(self, texts, dense):
        if dense == "st":
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "_st"):
                self._st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return _norm(self._st.encode(texts, batch_size=64, show_progress_bar=False))
        if dense == "azure":
            from openai import AzureOpenAI
            c = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                            api_key=os.environ["AZURE_OPENAI_API_KEY"],
                            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
            out = []
            for i in range(0, len(texts), 64):
                r = c.embeddings.create(model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                                        input=[t[:8000] for t in texts[i:i + 64]])
                out += [d.embedding for d in r.data]
            return _norm(np.array(out))
        raise ValueError(dense)

    def _embed_query(self, q):
        if self.dense_name == "lsa":
            return _norm(self._svd.transform(self._tfidf.transform([q])))[0]
        return self._encode([q], self.dense_name)[0]

    # ---- search --------------------------------------------------------------
    def search(self, query: str, k: int = 8, per_doc: int = 2, mode: str = "hybrid",
               exclude_docs: set | None = None) -> pd.DataFrame:
        """Top chunks, at most `per_doc` from any one report so the evidence is diverse.
        mode: 'hybrid' | 'bm25' | 'dense'."""
        n = len(self.chunks)
        ranks = {}
        if mode in ("hybrid", "bm25"):
            s = self.bm25.get_scores(tokenize(query))
            ranks["bm25"] = _rank(s)
        if mode in ("hybrid", "dense") and self.E is not None:
            s = self.E @ self._embed_query(query)
            ranks["dense"] = _rank(s)
        fused = np.zeros(n)
        for r in ranks.values():
            fused += 1.0 / (RRF_K + r)
        order = np.argsort(-fused)
        rows, per = [], {}
        for i in order:
            d = self.chunks.doc_id.iat[i]
            if exclude_docs and d in exclude_docs:
                continue
            if per.get(d, 0) >= per_doc:
                continue
            per[d] = per.get(d, 0) + 1
            rows.append(i)
            if len(rows) == k:
                break
        out = self.chunks.iloc[rows].copy()
        out["score"] = fused[rows]
        for name, r in ranks.items():
            out[f"{name}_rank"] = r[rows]
        meta = self.manifest.set_index("doc_id")
        out["title"] = out.doc_id.map(meta["title"])
        out["url"] = out.doc_id.map(meta["url"])
        return out.reset_index(drop=True)

    def top_docs(self, query: str, k: int = 5, mode: str = "hybrid") -> list[str]:
        hits = self.search(query, k=k * 3, per_doc=1, mode=mode)
        return list(dict.fromkeys(hits.doc_id))[:k]


def _rank(scores: np.ndarray) -> np.ndarray:
    r = np.empty(len(scores), dtype=int)
    r[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return r


def _norm(E):
    E = np.asarray(E, dtype=np.float32)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)


def _has_st() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
