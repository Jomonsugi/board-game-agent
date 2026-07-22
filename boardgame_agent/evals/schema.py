"""Dataset schema for offline evaluation.

The eval dataset is a single JSONL file — one ``EvalExample`` per line —
stored at ``boardgame_agent/evals/datasets/questions.jsonl``. All games live
in one file; each example carries a ``game_id`` and the runner filters by
game (``--games``). JSONL keeps appends cheap (e.g. promoting thumbs-up
answers from the UI) and diffs line-per-example in git.

Conventions:

- Gold answers and citations are verified against the **source PDFs**
  (rendered pages read visually), never against the extraction cache — the
  dataset describes ground truth, not current system behavior.
- ``gold_citations`` carry two page coordinates per citation:
  - ``page_num``: the page number a person sees — the label printed on the
    page, i.e. the page you would flip to in the physical rulebook. This is
    also the post-spread-split logical numbering (a spread-printed PDF page
    holds two printed pages). ``null`` when the page is unnumbered (e.g. a
    player aid).
  - ``pdf_page``: the physical 1-indexed page of the source PDF file.
  For most rulebooks the two are equal; they differ for spread-printed
  books and for books whose front matter offsets the printed numbering
  (printed = pdf − k).
  ``citation_page_hit`` counts a predicted (doc, page) as a hit if it
  matches either coordinate — tighten to one convention once the citation
  pipeline settles on it.
- ``tags`` drive filtered runs (``--tags icon``). Vocabulary:
  ``text`` (answerable from prose), ``icon`` (requires parsing iconography),
  ``multi-hop`` (requires cross-referencing pages/documents),
  ``negative`` (correct answer is "the rules don't cover this").
- ``difficulty``: ``easy`` | ``moderate`` | ``hard``. ``hard`` marks the
  multi-hop / icon-cross-reference questions this agent exists for.
- ``source_urls`` link the forum threads (BGG etc.) showing real players
  asking the question — evidence the question is realistic.
- ``needs_human_review: true`` marks machine-drafted examples whose gold
  answer has not been human-verified. The runner excludes them by default;
  include with ``--include-unreviewed``. Flip to false once verified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DATASETS_DIR = Path(__file__).parent / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "questions.jsonl"


class GoldCitation(BaseModel):
    doc_name: str
    page_num: int | None = Field(
        default=None,
        description=(
            "Page number as printed on the page / post-spread-split logical "
            "page — what a human flips to. None if the page is unnumbered."
        ),
    )
    pdf_page: int | None = Field(
        default=None,
        description="Physical 1-indexed page of the source PDF file.",
    )
    region: str = Field(
        default="",
        description=(
            "Human-readable location hint on the page (e.g. 'task-token "
            "table, top-right cell') — for review and future bbox grading."
        ),
    )

    def page_candidates(self) -> set[int]:
        return {p for p in (self.page_num, self.pdf_page) if p is not None}


class EvalExample(BaseModel):
    id: str = Field(description="Stable unique id, e.g. 'crew-icon-001'")
    game_id: str = Field(description="Folder name under data/games/")
    question: str
    gold_answer: str = Field(description="The correct answer, conversational but precise")
    gold_citations: list[GoldCitation] = Field(
        default_factory=list,
        description="Where the answer is written. Prediction hits if ANY gold citation matches.",
    )
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "moderate", "hard"] = "moderate"
    source_urls: list[str] = Field(
        default_factory=list,
        description="Forum threads showing real players asking this",
    )
    needs_human_review: bool = Field(
        default=False,
        description="True for machine-drafted examples pending human verification.",
    )
    notes: str = Field(default="", description="Rationale, edge cases, review comments")


def load_dataset(
    path: Path = DEFAULT_DATASET,
    games: list[str] | None = None,
) -> list[EvalExample]:
    """Load examples, optionally filtered to a list of game_ids."""
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
    if games:
        known = {e.game_id for e in examples}
        unknown = set(games) - known
        if unknown:
            raise ValueError(
                f"{path}: unknown game_id(s) {sorted(unknown)}; dataset has {sorted(known)}"
            )
        examples = [e for e in examples if e.game_id in games]
    return examples


def append_example(path: Path, example: EvalExample) -> None:
    """Append one example (e.g. promoted from a thumbs-up answer in the UI)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(example.model_dump_json() + "\n")
