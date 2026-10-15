"""Command line interface.

  python -m rag_core ingest   [--doc gpo|worldbank|imac|all] [--variant naive|advanced|all]
  python -m rag_core ask      --doc gpo "How many depository libraries did GPO staff visit in FY 2018?"
  python -m rag_core eval     [--doc all] [--variants naive,advanced] [--no-answers] [--judge]
  python -m rag_core inspect  --doc worldbank --page 96      (see how a page is parsed)
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

from . import documents, pipeline


def _docs(arg: str) -> list[str]:
    return list(documents.DOCUMENTS) if arg == "all" else [arg]


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("pymupdf").setLevel(logging.ERROR)
    ap = argparse.ArgumentParser(prog="rag_core")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--doc", default="all")
    p.add_argument("--variant", default="all", choices=["naive", "advanced", "all"])
    p.add_argument("--skip-existing", action="store_true", help="only build indexes that are missing")

    p = sub.add_parser("ask")
    p.add_argument("--doc", required=True)
    p.add_argument("--variant", default="advanced")
    p.add_argument("--trace", action="store_true")
    p.add_argument("question", nargs="+")

    p = sub.add_parser("eval")
    p.add_argument("--doc", default="all")
    p.add_argument("--variants", default="naive,advanced")
    p.add_argument("--no-answers", action="store_true", help="retrieval metrics only (no LLM calls)")
    p.add_argument("--judge", action="store_true", help="LLM-judged faithfulness")

    p = sub.add_parser("inspect")
    p.add_argument("--doc", required=True)
    p.add_argument("--page", type=int, required=True)

    a = ap.parse_args()
    if a.cmd == "ingest":
        variants = ("naive", "advanced") if a.variant == "all" else (a.variant,)
        for d in _docs(a.doc):
            print(f"== {d}")
            stats = pipeline.ingest(d, variants, skip_existing=a.skip_existing)
            print(json.dumps(stats, indent=1))
    elif a.cmd == "ask":
        rag = pipeline.build(a.doc, a.variant)
        res = rag.ask(" ".join(a.question), generate_answer=pipeline.can_generate())
        print(res.answer or "(no GEMINI_API_KEY: retrieval only)\n")
        print("\nContext passages:")
        for i, c in enumerate(res.contexts, 1):
            print(f"  [{i}] p.{c.get('pages')} {' > '.join(c.get('section_path') or [])} {c.get('label') or ''}")
        if a.trace:
            print(json.dumps({"trace": res.trace, "timings": res.timings}, indent=1, default=str))
    elif a.cmd == "eval":
        from . import evaluation
        for d in _docs(a.doc):
            print(f"== {d}")
            res = evaluation.evaluate(d, a.variants.split(","), with_answers=None if not a.no_answers else False,
                                      with_judge=a.judge)
            print(evaluation.to_markdown(res).split("## Per question")[0])
    elif a.cmd == "inspect":
        from .parsing.structure import parse_pdf
        cfg = documents.get(a.doc)
        parsed = parse_pdf(cfg.path, cfg.id)
        for e in parsed.elements:
            if e.page == a.page:
                print(f"[{e.kind}{' L' + str(e.level) if e.level else ''}{' ' + e.label if e.label else ''}] "
                      f"{' > '.join(e.section_path)}\n    {e.text[:300]}\n")


if __name__ == "__main__":
    main()
