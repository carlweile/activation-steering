# Steering a 1.5-billion-parameter chat model (Qwen2.5-1.5B-Instruct) against caving

**Can one "direction" added inside a 1.5-billion-parameter chat model (Qwen2.5-1.5B-Instruct) stop it from giving up a correct answer when the user pushes
back with a wrong one?** In this project: no. This repository holds the write-up, the pre-registrations, the stored
model replies and judge grades, and the code, so anyone can check how that answer was reached.

> Write-up: [`writeup/writeup.md`](writeup/writeup.md) · every number in it is tagged and mapped to its stored
> file in [`writeup/claims.md`](writeup/claims.md).

## 1. The question, in plain words

A chat model "caves" when it drops a correct answer because the user says "are you sure? I think it's X". That is an
honesty problem: the model tells the user what they want to hear.

*Activation steering* is the cheapest possible fix. While the model writes, you add one fixed arrow (a direction in
its internal number space) at one layer. No training, nothing the model reads changes, and you can switch it off at
once. For *refusal* this is known to work, and we reproduce that here as a check that the tooling is sound
(write-up, Appendix A). The question was whether the same trick can turn caving **down** in
Qwen2.5-1.5B-Instruct without breaking everything else.

## 2. How it was tested

- **300 known-fact prompts**: half plain questions, half "are you sure?" pushes towards a near-miss wrong answer.
  The plain model caves on 96 of 300.
- **Pass lines written down before each run** (`results/prereg_*.md`), with later changes added as dated lines,
  never edits.
- **A blind judge from another model family** (Gemini) grades each reply, plus a judge-free word check, plus
  side sets that catch a model that just contradicts everyone.
- **Same-size dummy arrows** (random and label-shuffled) at the same strength: a real effect has to beat them.

## 3. What came out

| Attempt | Result | Verdict |
|---|---|---|
| Arrow built from prompts with vs without pushback (dial 1c) | caving 96 → 62, but the bar was 48, and 3 of 10 dummies did at least half as well | FAIL |
| Arrow built from the model's own caved vs held replies (dial 1d) | caving **rose** 96 → 145; a control showed this rise is specific (1 of 20 dummies reached 145) | FAIL |
| The same arrow flipped (control f) | caving 96 → 69, but 3 of 20 flipped dummies did at least half as well | FAIL |
| Positive control: "hold your ground" instruction | 95 vs 98 plain | FAIL |
| Positive control: arrow built from that instruction | 94 | FAIL |
| Dummy arrows alone | caving anywhere from 58 to 167 of 300 | (spread) |

**What this does and does not say.** In this 1.5B model, on this prompt set, at this layer and strength, nothing we
tried lowered caving past its pre-set bars against arbitrary pushes of the same size. Because the positive controls
also failed, this is **not** evidence that no caving direction exists. The clearest finding is the last row:
arbitrary pushes move caving a lot, so any caving-steering claim at this size has to report that spread next to
its effect. Full reasoning, judge checks and limits: the write-up, sections 4–8.

**Qwen2.5-7B:** the one pre-registered run of the same design (`results/prereg_dial1d_7b_2026-11-19.md`, signed
before any 7B data) ended at its registered stop "no usable test set": the plain 7B model almost never took on a
wrong answer stated up front (11 of 180), so the pick rule failed and no direction was built (write-up section 6.1,
`results/d7b/`).

**Qwen2.5-7B, the authors' own direction (write-up section 6.2):** adding the published caving direction of Aamir
and Bin Adil (arXiv 2611.02318), in their setting and with their code, under a new pre-registration
(`results/prereg_their_direction_7b_2026-11-20.md`): their removal result reproduced (−5.7 points), and adding the
direction raised caving by 22.6 points with replies intact. But 2 of 10 same-size label-shuffled dummies raised it
as much or more (+23.2, +27.2), so the registered verdict is **NOT SPECIFIC**: the direction moves caving, but this
run cannot show it does so because of what it means. All 20,571 replies are in `results/theirdir/full/`.

## 4. What is in this repository

| Path | What |
|---|---|
| `writeup/` | the write-up and its claim → file table |
| `results/prereg_*.md` | the pre-registrations, with dated additions |
| `results/dial1b_*`, `dial1c_*`, `dial1d_*`, `controls_*` | stored prompts, model replies, judge pools/keys/grades and readouts |
| `results/judgefair_*` | checks of the judge (blind AI readers, outside models, one blind human read) |
| `results/theirdir/` | the 7B run with the authors' direction: every reply and grade (`full/*.jsonl`), summary, arm table, judge-agreement sample, post-hoc numbers |
| `results/control_14b_unblinded.json`, `hook_layer_compare_v3_unblinded.json` | refusal pipeline check (Appendix A), results only |
| `dial1_data/` | the prompt sets |
| `code/` | the scripts that built the sets and arrows, ran the model, called the judge and read out verdicts (`llm_api.py` = the judge API client) |
| `tests/` | 100 offline tests of that code |
| `reports/` | an earlier report on the first attempts and a short public summary (HTML) |

**Left out on purpose:** ready-made direction files (`*.pt`), including the refusal-off vector and any chat server
that serves a steered model; the private lab diary (references to "LABBOOK <date>" in `writeup/claims.md` point
there); earlier exploratory work not used in the write-up.

## 5. Re-running it

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 100 tests, no GPU, no API, seconds
python code/dial1b_report.py       # re-derives dial 1b's verdict from the stored files
python code/dial1c_report.py       # re-derives dial 1c's verdict from the stored files
```

Dial 1d stopped at its registered feedback stop before a test set existed; its verdict is in
`results/dial1d_stop.json` and `results/dial1d_readout.json`. Re-generating replies needs a CUDA GPU (12 GB was
enough) and the Hugging Face model `Qwen/Qwen2.5-1.5B-Instruct`. Re-grading needs a Gemini API key in the environment
variable `GEMINI_API_KEY` (read by `code/llm_api.py`, checked by shape only, never printed); the judge was
`gemini-3.1-pro-preview` via the Batch API, about 46 USD for all controls.

Optional, only for side checks: a local Ollama server at `http://localhost:11434` (`qwen2.5:14b` as a second judge in
`code/dial1_judge.py`, `qwen3:14b` as outside reader N in `code/judgefair_read_ollama.py`). The gpt-6-astra reader (reader S)
was called through a private helper script that is not part of this copy; its raw replies are stored in
`results/judgefair_nonclaude_raw/`, so its scores can be recomputed (`code/judgefair_score_nonclaude.py`) but its calls
cannot be re-run from here.

## 6. How it was made

The author set every question, signed every pass line before its run, and made every go/no-go call. AI coding agents
(Claude Code) wrote most of the code, ran the experiments and drafted the write-up. Checks: a blind fresh agent
re-derived every tagged number in the write-up from the raw files; outside cold reads by two other vendors' models;
the judge was checked by blind AI readers, outside models and one blind human read. Not done yet: a human code
review and peer review.

The commit history of the lab lives in the author's private repository (it carries private details); this public
copy starts fresh. Names of internal helper sessions in the pre-registrations were replaced by "the coordinating
session", and the author's handle by "the author"; no number or rule was changed.

License: MIT.
