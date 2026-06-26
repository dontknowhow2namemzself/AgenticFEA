#!/usr/bin/env python3
"""Vector store for the knowledge base (plan layers 7-8, RAG).

Indexes the reference solutions (by their `embed_text`) and the text knowledge
notes into a persistent ChromaDB collection, and retrieves candidates by semantic
similarity. Embeddings go through OpenRouter (see llm_config.embed_texts).

Role in the iron law: retrieval returns CANDIDATE ids only. The deterministic
`applies()` predicate in reference_solutions.py (used by verification.py) decides
which retrieved candidate is actually valid -- retrieval never picks a formula by
itself. See the module docstring of reference_solutions.py.

Offline self-test: `--selftest` uses an in-memory Chroma collection and a small
deterministic hash embedder, so the plumbing and rough ranking can be verified
WITHOUT an API key. The real-embedding path (`--build`, `--query`) needs the key.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROMA_DIR = HERE / "chroma"
COLLECTION = "fea_kb"

# Allow `from knowledge.* import ...` when run directly (python knowledge/vector_store.py).
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))


def _entries() -> list:
    """Gather everything to index: reference cases (by embed_text) + notes."""
    from knowledge.knowledge_notes import NOTES
    from knowledge.reference_solutions import KNOWLEDGE_BASE

    items = []
    for c in KNOWLEDGE_BASE:
        items.append({
            "id": c.id, "text": c.embed_text or c.applies_to,
            "metadata": {"kind": "reference", "name": c.name},
        })
    for n in NOTES:
        items.append({
            "id": n["id"], "text": n["text"],
            "metadata": {"kind": "note", "type": n["type"], "title": n["title"]},
        })
    return items


def _hash_embed(texts: list, dim: int = 384) -> list:
    """Deterministic bag-of-words hashing embedding (offline, no API). Shared words
    -> similar vectors, enough to exercise retrieval mechanics and rough ranking."""
    vecs = []
    for text in texts:
        v = [0.0] * dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            v[idx] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def _client(client=None):
    import chromadb
    return client or chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(client):
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index(client=None, embed_fn=None) -> int:
    """(Re)build the collection from the knowledge base. Returns the entry count."""
    embed_fn = embed_fn or _default_embed
    client = _client(client)
    coll = _collection(client)
    items = _entries()
    embeddings = embed_fn([it["text"] for it in items])
    coll.upsert(
        ids=[it["id"] for it in items],
        embeddings=embeddings,
        documents=[it["text"] for it in items],
        metadatas=[it["metadata"] for it in items],
    )
    return coll.count()


def retrieve(query: str, k: int = 4, kind: str = None, client=None, embed_fn=None) -> list:
    """Return the top-k knowledge entries for `query`, optionally filtered by
    kind ("reference" | "note"). Each result: {id, score, kind, text, metadata}."""
    embed_fn = embed_fn or _default_embed
    client = _client(client)
    coll = _collection(client)
    if coll.count() == 0:
        build_index(client=client, embed_fn=embed_fn)
    where = {"kind": kind} if kind else None
    res = coll.query(query_embeddings=embed_fn([query]), n_results=k, where=where)
    out = []
    ids = res["ids"][0]
    dists = (res.get("distances") or [[None] * len(ids)])[0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    for i, cid in enumerate(ids):
        score = (1.0 - dists[i]) if dists[i] is not None else None  # cosine similarity
        out.append({"id": cid, "score": score, "kind": metas[i].get("kind"),
                    "text": docs[i], "metadata": metas[i]})
    return out


def _default_embed(texts: list) -> list:
    import llm_config
    return llm_config.embed_texts(texts)


def _selftest() -> int:
    """Offline plumbing + rough-ranking check (in-memory chroma, hash embedder)."""
    import chromadb

    print("vector_store self-test (offline hash embedder, in-memory chroma)")
    print("=" * 64)
    client = chromadb.EphemeralClient()
    n = build_index(client=client, embed_fn=_hash_embed)
    print(f"indexed {n} entries (references + notes)")

    q = "cantilever beam fixed at one end with a transverse point load at the free end"
    refs = retrieve(q, k=4, kind="reference", client=client, embed_fn=_hash_embed)
    print(f"\nquery (references): {q}")
    for r in refs:
        print(f"  {r['score']:.3f}  {r['id']}")
    # The applies() gate re-checks every candidate, so the EXACT rank is not
    # critical for correctness; but for this clean query the point-load case
    # should clearly rank first.
    point_ok = refs[0]["id"] == "cantilever_tip_load_eb"

    fq = "the solver says the global stiffness matrix is singular"
    notes = retrieve(fq, k=3, kind="note", client=client, embed_fn=_hash_embed)
    print(f"\nquery (notes): {fq}")
    for r in notes:
        print(f"  {r['score']:.3f}  {r['id']}")
    note_ok = any(r["id"] == "fail_singular_stiffness" for r in notes)

    print("-" * 64)
    print(f"  [{'PASS' if point_ok else 'FAIL'}] point-load cantilever ranks #1 for its query")
    print(f"  [{'PASS' if note_ok else 'FAIL'}] singular-stiffness note retrieved for its query")
    ok = point_ok and note_ok
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}  (rough ranking via hash embedder; real "
          f"quality uses OpenRouter embeddings)")
    print("=" * 64)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Knowledge-base vector store (RAG, plan layer 7-8).")
    ap.add_argument("--build", action="store_true", help="(re)build the persistent index (needs API key)")
    ap.add_argument("--query", default=None, help="retrieve for a query (needs API key)")
    ap.add_argument("--kind", default=None, choices=["reference", "note"], help="filter by kind")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--selftest", action="store_true", help="offline plumbing test (no API key)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.build:
        n = build_index()
        print(f"built persistent index at {CHROMA_DIR} ({n} entries)")
        return 0
    if args.query:
        for r in retrieve(args.query, k=args.k, kind=args.kind):
            print(f"  {r['score']:.3f}  [{r['kind']}] {r['id']}")
        return 0
    ap.error("nothing to do: use --selftest, --build, or --query")


if __name__ == "__main__":
    sys.exit(main())
