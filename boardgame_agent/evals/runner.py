"""Headless eval runner.

Runs every dataset example through the agent graph (fresh thread per
question — no conversation bleed), judges answers with the configurable LLM
judge, computes citation-match metrics, and writes results + a summary to
``data/eval_runs/{timestamp}/``.

Usage:
    python -m boardgame_agent.evals.runner --game the_crew__the_quest_for_planet_nine
    python -m boardgame_agent.evals.runner --game <id> --tags icon           # subset
    python -m boardgame_agent.evals.runner --game <id> --model claude-sonnet-4-6
    python -m boardgame_agent.evals.runner --game <id> --include-unreviewed
    python -m boardgame_agent.evals.runner --game <id> --langsmith           # sync dataset + traces

Metrics:
- answer: correct / partial / incorrect (LLM judge, see judge.py)
- citation_doc_hit: any predicted citation names a gold doc
- citation_page_hit: any predicted citation matches a gold (doc, page) —
  this is the metric the bbox citation system uniquely enables; track it
  separately per tag (icon questions live or die on it)

LangSmith: per-run traces are captured automatically whenever
``LANGCHAIN_TRACING_V2=true`` (already wired in config.py). ``--langsmith``
additionally syncs the dataset to a LangSmith dataset named
``boardgame-{game_id}`` so runs can be inspected against examples in the UI.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from boardgame_agent.evals.schema import EvalExample, load_dataset

DATASETS_DIR = Path(__file__).parent / "datasets"


# ── Metrics ───────────────────────────────────────────────────────────────────

def citation_match(example: EvalExample, predicted_citations: list) -> dict:
    """Doc-level and page-level citation hits against gold citations."""
    if not example.gold_citations:
        return {"citation_doc_hit": None, "citation_page_hit": None}
    gold_docs = {g.doc_name for g in example.gold_citations}
    gold_pages = {(g.doc_name, g.page_num) for g in example.gold_citations}
    pred_docs = {c.doc_name for c in predicted_citations}
    pred_pages = {(c.doc_name, c.page_num) for c in predicted_citations}
    return {
        "citation_doc_hit": bool(gold_docs & pred_docs),
        "citation_page_hit": bool(gold_pages & pred_pages),
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evals(
    game_id: str,
    game_name: str,
    dataset_path: Path,
    model_name: str | None = None,
    judge_model: str | None = None,
    tags: list[str] | None = None,
    include_unreviewed: bool = False,
    limit: int | None = None,
    langsmith: bool = False,
) -> Path:
    """Run the eval suite; return the results directory."""
    from boardgame_agent.agent.graph import build_agent, run_query
    from boardgame_agent.config import DATA_DIR, DEFAULT_MODEL, EVAL_JUDGE_MODEL, EVAL_RUNS_DIR_NAME
    from boardgame_agent.evals.judge import build_judge

    examples = load_dataset(dataset_path)
    if not include_unreviewed:
        skipped = sum(1 for e in examples if e.needs_human_review)
        examples = [e for e in examples if not e.needs_human_review]
        if skipped:
            print(f"Skipping {skipped} unreviewed example(s) (--include-unreviewed to run them)")
    if tags:
        examples = [e for e in examples if set(tags) & set(e.tags)]
    if limit:
        examples = examples[:limit]
    if not examples:
        raise SystemExit("No examples to run after filtering.")

    model_name = model_name or DEFAULT_MODEL
    judge_model = judge_model or EVAL_JUDGE_MODEL
    print(f"Running {len(examples)} example(s) | agent={model_name} | judge={judge_model}")

    if langsmith:
        _sync_langsmith_dataset(game_id, examples)

    compiled, _llm, _client, _config = build_agent(game_id, game_name, model_name)
    judge = build_judge(judge_model)

    run_dir = DATA_DIR / EVAL_RUNS_DIR_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    rows: list[dict] = []
    with open(results_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            try:
                qa = run_query(compiled, game_id, ex.question, thread_id=f"eval-{uuid.uuid4()}")
                verdict = judge(ex.question, ex.gold_answer, qa.answer)
                row = {
                    "id": ex.id,
                    "question": ex.question,
                    "tags": ex.tags,
                    "agent_answer": qa.answer,
                    "gold_answer": ex.gold_answer,
                    "verdict": verdict.verdict,
                    "judge_reasoning": verdict.reasoning,
                    "predicted_citations": [c.model_dump() for c in qa.citations],
                    "gold_citations": [g.model_dump() for g in ex.gold_citations],
                    **citation_match(ex, qa.citations),
                    "confidence": qa.confidence,
                    "latency_s": round(time.time() - t0, 1),
                }
            except Exception as e:  # noqa: BLE001 — one bad question shouldn't kill the run
                row = {
                    "id": ex.id, "question": ex.question, "tags": ex.tags,
                    "verdict": "error", "error": f"{type(e).__name__}: {e}",
                    "citation_doc_hit": None, "citation_page_hit": None,
                    "latency_s": round(time.time() - t0, 1),
                }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(examples)}] {ex.id}: {row['verdict']}"
                  + (f" (page_hit={row['citation_page_hit']})"
                     if row.get("citation_page_hit") is not None else ""))

    summary = _summarize(rows, model_name, judge_model, dataset_path)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(summary)
    print(f"\nResults: {results_path}")
    return run_dir


def _summarize(rows: list[dict], model_name: str, judge_model: str, dataset_path: Path) -> dict:
    def rate(rs: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def block(rs: list[dict]) -> dict:
        verdicts = Counter(r["verdict"] for r in rs)
        return {
            "n": len(rs),
            "correct": verdicts.get("correct", 0),
            "partial": verdicts.get("partial", 0),
            "incorrect": verdicts.get("incorrect", 0),
            "error": verdicts.get("error", 0),
            "correct_rate": round(verdicts.get("correct", 0) / len(rs), 3),
            "citation_doc_hit_rate": rate(rs, "citation_doc_hit"),
            "citation_page_hit_rate": rate(rs, "citation_page_hit"),
        }

    all_tags = sorted({t for r in rows for t in r.get("tags", [])})
    return {
        "model": model_name,
        "judge_model": judge_model,
        "dataset": str(dataset_path),
        "overall": block(rows),
        "by_tag": {t: block([r for r in rows if t in r.get("tags", [])]) for t in all_tags},
        "mean_latency_s": rate(rows, "latency_s"),
    }


def _print_summary(s: dict) -> None:
    o = s["overall"]
    print(f"\n{'─' * 60}")
    print(f"OVERALL  n={o['n']}  correct={o['correct']}  partial={o['partial']}  "
          f"incorrect={o['incorrect']}  errors={o['error']}")
    print(f"  correct_rate={o['correct_rate']}  "
          f"doc_hit={o['citation_doc_hit_rate']}  page_hit={o['citation_page_hit_rate']}")
    for tag, b in s["by_tag"].items():
        print(f"  [{tag:>10}] n={b['n']:<3} correct_rate={b['correct_rate']}  "
              f"page_hit={b['citation_page_hit_rate']}")


def _sync_langsmith_dataset(game_id: str, examples: list[EvalExample]) -> None:
    """Create/refresh the LangSmith dataset ``boardgame-{game_id}``."""
    try:
        from langsmith import Client
        client = Client()
        name = f"boardgame-{game_id}"
        try:
            dataset = client.read_dataset(dataset_name=name)
        except Exception:  # noqa: BLE001
            dataset = client.create_dataset(dataset_name=name)
        existing = {
            e.metadata.get("example_id")
            for e in client.list_examples(dataset_id=dataset.id)
            if e.metadata
        }
        new = [e for e in examples if e.id not in existing]
        if new:
            client.create_examples(
                dataset_id=dataset.id,
                inputs=[{"question": e.question} for e in new],
                outputs=[{"gold_answer": e.gold_answer,
                          "gold_citations": [g.model_dump() for g in e.gold_citations]}
                         for e in new],
                metadata=[{"example_id": e.id, "tags": e.tags} for e in new],
            )
        print(f"LangSmith: dataset '{name}' synced ({len(new)} new example(s))")
    except Exception as e:  # noqa: BLE001 — never fail an eval run on upload
        print(f"LangSmith sync skipped: {type(e).__name__}: {e}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline eval suite.")
    parser.add_argument("--game", required=True, help="game_id (folder under data/games/)")
    parser.add_argument("--game-name", default=None, help="Display name (defaults to game_id)")
    parser.add_argument("--dataset", default=None,
                        help=f"Dataset path (default: {DATASETS_DIR}/{{game}}.jsonl)")
    parser.add_argument("--model", default=None, help="Agent model (default: config DEFAULT_MODEL)")
    parser.add_argument("--judge-model", default=None, help="Judge model (default: config EVAL_JUDGE_MODEL)")
    parser.add_argument("--tags", nargs="*", default=None, help="Only run examples with these tags")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Include examples flagged needs_human_review")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--langsmith", action="store_true", help="Sync dataset to LangSmith")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else DATASETS_DIR / f"{args.game}.jsonl"
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    run_evals(
        game_id=args.game,
        game_name=args.game_name or args.game.replace("_", " ").title(),
        dataset_path=dataset_path,
        model_name=args.model,
        judge_model=args.judge_model,
        tags=args.tags,
        include_unreviewed=args.include_unreviewed,
        limit=args.limit,
        langsmith=args.langsmith,
    )


if __name__ == "__main__":
    _main()
