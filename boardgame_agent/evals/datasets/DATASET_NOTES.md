# Eval dataset — verification notes

`questions.jsonl` is the unified gold dataset: 18 examples, 6 games, 3 per game
(1–2 hard multi-hop/icon questions + 1 moderate). Every gold answer and page
reference was verified against the **source PDFs** in `data/games/*/docs/` by
rendering the cited pages to images and reading them visually (pdftoppm @
120–150 dpi). The extraction cache (`extracted/`) was deliberately never used —
the dataset describes ground truth, not current system behavior. Questions were
sourced from real player threads (BGG, EN World, D&D Beyond, ultraboardgames);
`source_urls` on each example link the evidence that people actually ask them.

Replaces `the_crew__the_quest_for_planet_nine.jsonl` (single-game, superseded —
safe to delete).

## Page-numbering conventions per document

Each gold citation stores `page_num` (the number printed on the page — what you
flip to in the physical book; also the post-spread-split logical page) and
`pdf_page` (physical 1-indexed page of the PDF file). Verified mappings:

| Document | Mapping |
|---|---|
| Crew Rules | printed = pdf (22 pages, offset 0) |
| Crew Log Book | **spreads**: pdf page N holds printed pages 2N−2 / 2N−1 (pdf 1 = unnumbered cover) |
| Player's Handbook (2014, 1st printing) | printed = pdf − 1 |
| Monster Manual (2014, 1st printing) | printed = pdf − 1 |
| Dungeon Master's Guide (2014, 1st printing) | printed = pdf (scan omits cover) |
| Gloomhaven_rules | printed = pdf (52 pages) |
| grandaustriahotel_rules (2022 Lookout revised EN) | printed = pdf (20 pages) |
| skyteam_rulebook / skyteam_flight_log | printed = pdf (12 / 8 pages) |
| LotR DfME rules | printed = pdf (8 pages, medallion at bottom-center) |
| LotR DfME player aid | **no printed page numbers** → `page_num: null` |

`citation_page_hit` in the runner accepts a predicted page matching either
coordinate; tighten to one convention once the citation pipeline settles.

## Corrections vs. the old Crew draft

- **Omega (Ω) is not in mission 9.** It appears in missions 7, 12, and 28
  (mission 9 is "1 Trick with" 1-value cards). Ω = "the task must be fulfilled
  last" — last of the pending tasks, *not* necessarily the last trick (Rules
  p14 + p10; the mission ends the moment all tasks are done).
- **The dead zone symbol is the green radio token + white "?"** (Rules p17),
  not a crossed-out radio. The crossed-out radio (red X) is mission 11's
  mission-specific art. Disruption is the token + red lightning bolt + trick
  number.
- **Old page refs pointed at physical logbook PDF pages**; the logbook is
  scanned as spreads, so those never matched what a player (or the
  spread-splitting extractor) sees. Citations now carry both coordinates.

## Spot-verification highlights (what was checked visually)

- Crew: logbook pdf p4 (missions 7/9/12 + Ω tiles), pdf p6 (missions 19/20/21
  icons), pdf p7 (golden frames on 25/27/28/30), legend on pdf p2; Rules p14
  (task-token table), p17 (dead zone / disruption / commander's decision),
  p18–19 (five-player rule, "missions 27 and 37", no extra attempt).
- D&D: edition confirmed from credits pages (2014 first printings, ISBNs);
  Counterspell (PHB printed 228), Fire Breath in Adult Red Dragon Actions
  (MM printed 98), Cover (PHB 196), Hitting Cover optional rule (DMG 272),
  Bonus Action casting (PHB 202), offsets checked on 3 pages per book.
- Gloomhaven: trap trigger + jump/fly exemption (p14), jump last-hex rule
  (p19), elemental infusion timing (p23–24 incl. the bolded same-turn
  prohibition), PIERCE worked example (p22), retaliate (p26).
- Grand Austria: emperor slash-penalty rule (p11) + A-tile text (p20), icon
  legend (p16), occupancy bonus (p9), imitate-action example (p8). 2022
  revised rulebook (objective cards = old "politics cards").
- Sky Team: module rules live in the *flight log* (pp3–5), not the rulebook;
  scenario→icon map on pp6–8 verified (OSL red = kerosene-leak + ice-brakes
  icons, GIG yellow = windsock, etc.); coffee cap + reserve (rulebook p8, p3).
- LotR DfME: quest-track bonus icons used on rules p6, defined in player aid
  p1 with one-time-use footnote; Eagle scoped to Support of the Races (aid p2
  + rules p8 note) vs Green-card-only alliance triggers (rules p6); landmark
  skill-substitution + fortress surcharge (rules p4).

## Auditing

To audit any example: open the PDF at `pdf_page`, look at `region`, and check
the quote-level evidence in each example's `notes`. The `source_urls` show the
question is realistic, not that the answer is right — answers were taken from
the rulebooks themselves.
