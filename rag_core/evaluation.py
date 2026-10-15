"""Evaluation: compare pipeline variants on a fixed question set per document.

Retrieval metrics (no LLM needed):
  hit@k           the context sent to the LLM includes an expected page
  MRR             1 / rank of the first context passage from an expected page
  context recall  share of required facts present anywhere in the context (did the facts reach the LLM?)
Answer metrics (need GEMINI_API_KEY):
  accuracy        every required fact appears in the answer
  refusal         unanswerable questions get "does not contain"; answerable ones are not refused
  faithfulness    (optional, --judge) an LLM judge checks every claim is supported by the context
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime

from . import config, generate, llm, pipeline


def _contains(text: str, alternatives: list[str]) -> bool:
    t = text.lower().replace(" ", " ")
    return any(a.lower() in t for a in alternatives)


JUDGE_SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"},
                                                  "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
                "required": ["supported"]}


def judge(question: str, answer: str, contexts: list[dict]) -> dict:
    system = ("You check whether an answer is fully supported by the given context passages. "
              "supported = true only if every factual claim in the answer is stated in the context. "
              "An answer saying the information is not in the document counts as supported.")
    prompt = f"Context:\n{generate.format_context(contexts)}\n\nQuestion: {question}\nAnswer: {answer}"
    return llm.generate_json(prompt, system, JUDGE_SCHEMA, model=config.JUDGE_MODEL, cache="judge")


def evaluate(doc_id: str, variants: list[str], with_answers: bool | None = None, with_judge: bool = False,
             progress=print) -> dict:
    questions = json.loads((config.EVAL_DIR / "questions" / f"{doc_id}.json").read_text())
    with_answers = llm.has_key() if with_answers is None else with_answers
    results = {"doc": doc_id, "date": datetime.now().isoformat(timespec="seconds"),
               "embedder": pipeline.embeddings.dense().name, "answers": with_answers, "variants": {}}

    for variant in variants:
        rag = pipeline.build(doc_id, variant)
        rows = []
        for q in questions:
            answerable = q.get("answerable", True)
            t = time.perf_counter()
            res = rag.ask(q["question"], generate_answer=with_answers)
            row = {"id": q["id"], "question": q["question"], "source": q.get("source"),
                   "answerable": answerable, "latency": round(time.perf_counter() - t, 2),
                   "context_pages": [c.get("pages") or [c.get("page")] for c in res.contexts]}
            if answerable:
                expected = set(q["pages"])
                ranks = [i + 1 for i, c in enumerate(res.contexts)
                         if expected & set(c.get("pages") or [c.get("page")])]
                ctx_text = " ".join(c["text"] for c in res.contexts)
                row.update(hit=bool(ranks), rr=1 / ranks[0] if ranks else 0.0,
                           context_recall=sum(_contains(ctx_text, g) for g in q["facts"]) / len(q["facts"]))
            if with_answers:
                refused = generate.NOT_FOUND.lower()[:30] in res.answer.lower() or "does not contain" in res.answer.lower()
                row["answer"] = res.answer
                if answerable:
                    row["correct"] = (not refused) and all(_contains(res.answer, g) for g in q["facts"])
                else:
                    row["correct"] = refused
                if with_judge:
                    row["faithful"] = judge(q["question"], res.answer, res.contexts).get("supported")
            rows.append(row)
            progress(f"  [{variant}] {q['id']}: " + ", ".join(
                f"{k}={row[k]:.2f}" if isinstance(row.get(k), float) else f"{k}={row[k]}"
                for k in ("hit", "context_recall", "correct") if k in row))
        results["variants"][variant] = {"rows": rows, "summary": summarize(rows)}

    out = config.EVAL_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{doc_id}.json").write_text(json.dumps(results, indent=1))
    (out / f"{doc_id}.md").write_text(to_markdown(results))
    return results


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    s = {
        "questions": len(rows),
        "hit@k": statistics.mean(r["hit"] for r in ans) if ans else None,
        "mrr": statistics.mean(r["rr"] for r in ans) if ans else None,
        "context_recall": statistics.mean(r["context_recall"] for r in ans) if ans else None,
        "avg_latency_s": round(statistics.mean(r["latency"] for r in rows), 2),
    }
    if rows and "correct" in rows[0]:
        s["accuracy"] = statistics.mean(r["correct"] for r in rows)
    if rows and "faithful" in rows[0]:
        s["faithfulness"] = statistics.mean(bool(r["faithful"]) for r in rows)
    return s


def to_markdown(results: dict) -> str:
    variants = results["variants"]
    metrics = [m for m in ("hit@k", "mrr", "context_recall", "accuracy", "faithfulness", "avg_latency_s")
               if any(m in v["summary"] for v in variants.values())]
    lines = [f"# Evaluation: {results['doc']}", "",
             f"Run {results['date']} · embeddings `{results['embedder']}` · answers generated: {results['answers']}", "",
             "| Variant | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for name, v in variants.items():
        cells = []
        for m in metrics:
            val = v["summary"].get(m)
            cells.append("—" if val is None else (f"{val:.0%}" if m not in ("mrr", "avg_latency_s") else f"{val:.2f}"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Per question", ""]
    first = next(iter(variants.values()))["rows"]
    header = "| Question | " + " | ".join(variants) + " |"
    lines += [header, "|---|" + "---|" * len(variants)]
    for i, row in enumerate(first):
        cells = []
        for v in variants.values():
            r = v["rows"][i]
            if "correct" in r:
                cells.append("✅" if r["correct"] else "❌")
            elif r["answerable"]:
                cells.append(f"hit {'✓' if r['hit'] else '✗'} · facts {r['context_recall']:.0%}")
            else:
                cells.append("n/a")
        lines.append(f"| {row['question']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
