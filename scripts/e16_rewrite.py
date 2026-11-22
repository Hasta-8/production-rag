"""E16: query-rewrite v1 vs v2 on vague questions and on every trap question.

uv run python scripts/e16_rewrite.py   (about 30 minutes on the laptop CPU)
Writes eval/results/<date>_e16-v1.json and _e16-v2.json.
"""

import dataclasses
import json
import time
from datetime import date
from pathlib import Path

import yaml

from rag.agent import RAGAgent
from rag.config import get_settings
from rag.runtime import answer_settings, make_llm, make_retriever, open_index

HELDOUT_TEST = {"vh1", "vh2", "vh3", "vh6", "vh9", "vh10"}  # gate failures (retrieval-only check)


def questions():
    out = []
    for q in yaml.safe_load(Path("eval/vague.yaml").read_text())["questions"]:
        if q["split"] == "tune" or q["id"] in HELDOUT_TEST:
            out.append({**q, "kind": f"vague-{q['split']}"})
    for name in ("golden", "heldout", "heldout2"):
        for q in yaml.safe_load(Path(f"eval/{name}.yaml").read_text())["questions"]:
            if q.get("answerable") is False:
                out.append({"id": q["id"], "question": q["question"], "kind": "trap"})
    return out


def main():
    s = get_settings()
    index, _ = open_index(s)
    retriever = make_retriever(s, index)
    llm = make_llm(s)
    arxiv = {
        m["doc_id"]: str(m.get("arxiv_id"))
        for m in index.store.get(include=["metadatas"])["metadatas"]
    }
    qs = questions()
    for version in ("1", "2"):
        opts = dataclasses.replace(answer_settings(s), rewrite_version=version)
        agent = RAGAgent(retriever, llm, opts)
        rows = []
        for q in qs:
            t0 = time.perf_counter()
            a = agent.answer(q["question"])
            steps = a.trace.steps
            cited = [arxiv.get(src.doc_id) for src in a.cited_sources]
            row = {
                "id": q["id"],
                "kind": q["kind"],
                "question": q["question"],
                "abstained": a.abstained,
                "reason": a.abstain_reason,
                "retries": a.retries,
                "rewrites": [st.data.get("query") for st in steps if st.name == "rewrite"],
                "grades": [st.data.get("best_score") for st in steps if st.name == "grade"],
                "answer": a.text,
                "cited_arxiv": cited,
                "right_paper": (q.get("arxiv_id") in cited) if "arxiv_id" in q else None,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "v": version,
                        **{
                            k: row[k]
                            for k in ("id", "abstained", "retries", "rewrites", "right_paper")
                        },
                    }
                ),
                flush=True,
            )
        test = [r for r in rows if r["kind"] == "vague-heldout"]
        traps = [r for r in rows if r["kind"] == "trap"]
        summary = {
            "heldout_rescued": sum(not r["abstained"] and r["right_paper"] for r in test),
            "heldout_n": len(test),
            "tune_rescued": sum(
                not r["abstained"] and r["right_paper"] for r in rows if r["kind"] == "vague-tune"
            ),
            "traps_declined": sum(r["abstained"] for r in traps),
            "traps_n": len(traps),
            "traps_answered_after_rewrite": [
                r["id"] for r in traps if not r["abstained"] and r["retries"]
            ],
        }
        out = Path(f"eval/results/{date.today().isoformat()}_e16-v{version}.json")
        out.write_text(json.dumps({"version": version, "summary": summary, "rows": rows}, indent=2))
        print("SUMMARY", version, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
