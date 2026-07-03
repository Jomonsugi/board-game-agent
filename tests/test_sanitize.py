"""Tests for VLM description sanitization.

Failure strings below are verbatim from The Crew logbook extraction
(Qwen2.5-VL 3B via Docling picture description).
"""

import json

import pytest

from boardgame_agent.rag.sanitize import (
    retro_sanitize_game,
    sanitize_page,
    sanitize_vlm_description,
)


# ── sanitize_vlm_description ──────────────────────────────────────────────────

GOOD = [
    (
        "The image shows a red star with the number 2 inside it.<|im_",
        "The image shows a red star with the number 2 inside it.",
    ),
    (
        "The image shows a black pentagon with the number \"1\" inside it.<|im_end|>",
        "The image shows a black pentagon with the number \"1\" inside it.",
    ),
    # Already clean — untouched.
    (
        "The image shows a blue square icon with a white circle inside it.",
        "The image shows a blue square icon with a white circle inside it.",
    ),
]

REFUSALS = [
    "I'm sorry, but I cannot see any image attached to your question. Please provide an image for me to describe.<|",
    "I'm sorry, but I cannot describe the image as there is no image provided. Please provide an image for me to analyze.",
    "There is no image attached to this message.",
    "I am unable to see the image you are referring to.",
    "As an AI text model, I cannot view images.",
    # Refusal + template leakage combined.
    "I'm sorry, but I cannot see any image attached.<|im_end|>",
]

NOISE = ["", None, "A.", "img", "   ", "<|im_end|>"]


@pytest.mark.parametrize("dirty,expected", GOOD)
def test_strips_template_tokens_keeps_content(dirty, expected):
    assert sanitize_vlm_description(dirty) == expected


@pytest.mark.parametrize("refusal", REFUSALS)
def test_refusals_become_empty(refusal):
    assert sanitize_vlm_description(refusal) == ""


@pytest.mark.parametrize("noise", NOISE)
def test_noise_becomes_empty(noise):
    assert sanitize_vlm_description(noise) == ""


def test_legit_description_mentioning_images_survives():
    # "image" appearing in a real description must not trigger refusal logic.
    text = "The image shows four playing cards with the number \"1\" on them."
    assert sanitize_vlm_description(text) == text


# ── sanitize_page ─────────────────────────────────────────────────────────────

def _make_page():
    refusal = "I'm sorry, but I cannot see any image attached to your question.<|"
    leaked = "The image shows a red star with the number 2 inside it.<|im_"
    clean = "Here is the mission text."
    return {
        "game_id": "g",
        "doc_name": "d",
        "page_num": 4,
        "text": "\n\n".join([clean, refusal, leaked]),
        "bboxes": [
            {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "text": clean, "label": "text"},
            {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "text": refusal, "label": "picture"},
            {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "text": leaked, "label": "picture"},
        ],
    }


def test_sanitize_page_cleans_bboxes_and_page_text():
    page = _make_page()
    changed = sanitize_page(page)
    assert changed == 2
    assert page["bboxes"][1]["text"] == ""
    assert page["bboxes"][2]["text"] == "The image shows a red star with the number 2 inside it."
    assert "sorry" not in page["text"]
    assert "<|" not in page["text"]
    assert "Here is the mission text." in page["text"]
    # Bbox coordinates untouched — citations still resolve.
    assert page["bboxes"][1]["x1"] == 1


def test_sanitize_page_idempotent():
    page = _make_page()
    sanitize_page(page)
    assert sanitize_page(page) == 0


def test_non_picture_bboxes_never_touched():
    page = _make_page()
    page["bboxes"][0]["text"] = "I'm sorry, but I cannot see any image attached."  # real OCR'd text
    sanitize_page(page)
    assert page["bboxes"][0]["text"] == "I'm sorry, but I cannot see any image attached."


# ── retro_sanitize_game ───────────────────────────────────────────────────────

def test_retro_sanitize_game(tmp_path):
    extracted = tmp_path / "games" / "mygame" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "Rules.json").write_text(json.dumps([_make_page()]))

    report = retro_sanitize_game("mygame", data_dir=tmp_path)
    assert report == {"Rules": 2}

    pages = json.loads((extracted / "Rules.json").read_text())
    assert pages[0]["bboxes"][1]["text"] == ""

    # Second pass: nothing left to clean.
    assert retro_sanitize_game("mygame", data_dir=tmp_path) == {"Rules": 0}


def test_retro_sanitize_dry_run(tmp_path):
    extracted = tmp_path / "games" / "mygame" / "extracted"
    extracted.mkdir(parents=True)
    original = json.dumps([_make_page()])
    (extracted / "Rules.json").write_text(original)

    report = retro_sanitize_game("mygame", data_dir=tmp_path, dry_run=True)
    assert report == {"Rules": 2}
    assert (extracted / "Rules.json").read_text() == original


def test_retro_sanitize_missing_game(tmp_path):
    with pytest.raises(FileNotFoundError):
        retro_sanitize_game("nope", data_dir=tmp_path)
