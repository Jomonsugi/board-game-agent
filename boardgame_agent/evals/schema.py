"""Dataset schema for offline evaluation.

Datasets are JSONL files, one ``EvalExample`` per line, stored in
``boardgame_agent/evals/datasets/{game_id}.jsonl``. JSONL was chosen so
examples can be appended (e.g. from thumbs-up answers in the UI) without
rewriting the file, and diffs stay line-per-example in git.

Conventions:

- ``gold_citations`` use *logical* page numbers — the same post-spread-split
  numbering the agent cites (matches ``page_num`` in the extraction cache).
- ``tags`` drive filtered runs (``--tags icon``). Suggested vocabulary:
  ``text`` (answerable from prose), ``icon`` (requires iconography),
  ``multi-hop`` (requires cross-referencing documents/sections),
  ``negative`` (correct answer is "the rules don't cover this").
- ``needs_human_review: true`` marks machine-drafted examples whose gold
  answer has not been human-verified. The runner excludes them by default;
  include with ``--include-unreviewed``. Flip the flag to false once verified.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldCitation(BaseModel):
    doc_name: str
    page_num: int


class EvalExample(BaseModel):
    id: str = Field(description="Stable unique id, e.g. 'crew-012'")
    question: str
    gold_answer: str = Field(description="The correct answer, concise but complete")
    gold_citations: list[GoldCitation] = Field(
        default_factory=list,
        description="Where the answer is written. Prediction hits if ANY gold citation matches.",
    )
    tags: list[str] = Field(default_factory=list)
    needs_human_review: bool = Field(
        default=False,
        description="True for machine-drafted examples pending human verification.",
    )
    notes: str = Field(default="", description="Rationale, edge cases, review comments")


def load_dataset(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                examples.append(EvalExample(**json.loads(line)))
            except Exception as e:  # noqa: BLE001 — surface line number
                raise ValueError(f"{path}:{line_no}: {e}") from e
    ids = [e.id for e in examples]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate example ids: {sorted(dupes)}")
    return examples


def append_example(path: Path, example: EvalExample) -> None:
    """Append one example (e.g. promoted from a thumbs-up answer in the UI)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(example.model_dump_json() + "\n")
