"""LLM judge for answer correctness.

The judge compares the agent's answer against the gold answer for a given
question and returns a structured verdict. Model is configurable via
``EVAL_JUDGE_MODEL`` (env var or config.py); defaults to Claude when an
Anthropic key exists — a strong judge decorrelated from the Together-served
agent models — otherwise falls back to the default agent model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from boardgame_agent.config import EVAL_JUDGE_MODEL


class JudgeVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description=(
            "correct: agent answer agrees with the gold answer on every point "
            "that matters for playing the game correctly. "
            "partial: agrees on the core ruling but omits or muddles a "
            "material detail. "
            "incorrect: contradicts the gold answer or fails to answer."
        )
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict")


_JUDGE_SYSTEM = """You are grading a board-game rules assistant.

Compare the AGENT ANSWER to the GOLD ANSWER for the QUESTION.
Judge only rules substance: does a player following the agent answer play the
game the same way as a player following the gold answer?

- Extra correct detail is fine; do not penalize verbosity or tone.
- Different wording for the same ruling is correct.
- A hedge ("the rulebook doesn't specify") when the gold answer gives a
  concrete ruling is incorrect.
- If the gold answer says the rules don't cover something, the agent must say
  the same to be correct — a confident invented ruling is incorrect."""

_JUDGE_USER = """QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

AGENT ANSWER:
{agent_answer}"""


def build_judge(model_name: str | None = None):
    """Return a callable (question, gold_answer, agent_answer) -> JudgeVerdict."""
    from boardgame_agent.agent.graph import _build_llm

    llm = _build_llm(model_name or EVAL_JUDGE_MODEL)
    structured = llm.with_structured_output(JudgeVerdict)

    def judge(question: str, gold_answer: str, agent_answer: str) -> JudgeVerdict:
        prompt = [
            ("system", _JUDGE_SYSTEM),
            ("user", _JUDGE_USER.format(
                question=question, gold_answer=gold_answer, agent_answer=agent_answer
            )),
        ]
        return structured.invoke(prompt)

    return judge
