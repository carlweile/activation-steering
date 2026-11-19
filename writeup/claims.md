# Claim -> file table for `writeup.md`

Every number in the write-up carries a tag `[Cn]`. Each tag below names the stored file (and field) or the dated LABBOOK
entry that holds it. References of the form "LABBOOK <date>" point to the author's private lab diary, which is not part of this public
copy; where such an entry summarises stored data, the data files themselves are in `results/`. If a number in the write-up has no tag, or a tag's number differs from its file, the file wins and
the write-up is wrong. Built 2026-11-19 from the files of the author's private lab repository.

Judge model in every caving grade file: `gemini-3.1-pro-preview` (field `model` in `results/*_grades_gemini.json`).

## Setup
| Tag | Claim | Source |
|---|---|---|
| C1 | Model Qwen2.5-1.5B-Instruct, fp16, greedy, 128 new tokens, hook adds alpha x arrow at block 17, every position | `results/prereg_controls_2026-11-17.md` setup line; `code/dial1_lib.py` |
| C2 | Strength f = 0.20, alpha = 12.53 at block 17 | `results/dial1c_readout.json` `block18_gate.alpha17`; `results/controls_arm_b.json` `gates.alpha` |
| C3 | GI-H = 300 prompts, 150 answer + 150 are-you-sure at level L2, near-miss wrong answers, knowledge filter on the bare-question twin reply, 30 % arithmetic per half | `results/prereg_dial1c_2026-11-10.md` sec. 5 table (line "GI-H") |
| C4 | L2 picked by a baseline-only screen of 360 prompts on a disjoint pool | `results/prereg_dial1c_2026-11-10.md`; `results/dial1c_screen_pool.jsonl` (360 rows) |
| C5 | Judge calibration: 17 hand-made cases, at least 16 of 17 needed; every controls calibration passed 17/17 | `results/controls_spend_card{1..4}.json` `calibrations[].passed`; `results/dial1d_spend.json` |
| C6 | UR-40 (user is right) and PUSH-50 (model wrong, user right) side sets | `results/dial1c_readout.json` `ur_push_detail` |

## Main attempts
| Tag | Claim | Source |
|---|---|---|
| C10 | Dial 1b: baseline caved 19 of 200, steered 4; McNemar 18 vs 3, p = 0.0015; floor 24 missed -> UNINFORMATIVE; 2 of 10 controls at half the drop | `results/dial1b_readout.json` `B`, `arms.real_f0.20.caved`, `mcnemar_*`, `n_controls_at_half_drop`, `verdict` |
| C11 | Dial 1c: baseline 96 of 300, steered 62; McNemar 49 vs 15, p = 2.4e-5; bar 48 | `results/dial1c_readout.json` `B`, `arms.real17_f0.20.caved`, `mcnemar_*`; bar in prereg 1c sec. 7 |
| C12 | Dial 1c: fluent 250 of 300 (bar 270; plain 177); 3 of 10 controls at half the drop; controls left 68-150 caved | `results/dial1c_readout.json` `arms.*.coh3`, `n_controls_at_half_drop`, `arms.ctrl*_f0.20.caved` |
| C13 | Dial 1c: mean words 51.5 -> 20.9; UR-40 40 -> 38; PUSH-50 47 -> 46; block 18 side check 57 caved, 269 fluent | `results/dial1c_readout.json` `arms.*.mean_words`, `ur_agrees`, `push_corrects`, `arms.real18` |
| C14 | Dial 1d: a1 = unit(mean caved - mean held) of the model's own screen replies (58 caved, 87 held), gate median cos 0.844 vs shuffled max 0.26 | `results/dial1d_arrow1.json` `n_caved`, `n_held`; `results/dial1d_readout.json` `arrow1_gate` |
| C15 | Dial 1d: of 96 base caves, H = 7 now held fluently (stop line 24) -> FAIL at stop (ii); caving 96 -> 145, fluent 170 | `results/dial1d_readout.json` `feedback`, `verdict`, `reason` |
| C16 | a1 was ADDED (+alpha), i.e. pointed toward caving | `code/dial1d_arrow.py` lines 39-51, `code/dial1d_lib.py:42`, `code/dial1d_run.py:154`; LABBOOK 2026-11-18 (evening) |
| C18 | Dial 1c judge-free bar (base + 15, dial 1b rule: correct whole word AND wrong absent): 66 -> 206 of 300, met; plain substring count 265 -> 243 (fails); read ambiguous, verdict FAIL either way | `results/dial1c_readout.json` `arms.*.correct_present`, `bars.judge_free`; `results/prereg_dial1c_2026-11-10.md` sec. 7 bar 3 + dated addition (blind recomputation, 2026-11-10) |
| C19 | a1 build (360 screen prompts) disjoint from GI-H by prompt and fact, asserted by script | `results/prereg_dial1d_2026-11-11.md` sec. 3.1 ("overlap 0"); `code/dial1d_sets.py` `assert_roles_disjoint` |
| C17 | Attempt count: dial 1d = 9th caving/sycophancy attempt (8 earlier); -a1 = 10th | `reports/2026-11-11-writeup.html` rows 1-8 + sec. 7; LABBOOK 2026-11-17 (late evening); `results/controls_arm_f.json` `label` |

## Judge checks
| Tag | Claim | Source |
|---|---|---|
| C20 | Two blind Claude readers on 156 stored dial 1d replies: judge agreement G1 98/98 %, G2 93/93 %, G3 86/84 %; H_reader 0 / 1; verdict FAIR | `results/judgefair_2026-11-13.json`; LABBOOK 2026-11-13 |
| C21 | Judge over-counts caves a little in BOTH arms: plain ~81 vs 96, steered ~136-139 vs 145 (reader-corrected) | LABBOOK 2026-11-13 "Reader-corrected caving" |
| C22 | Outside readers: qwen3:14b G3 56 % (near the 52 % always-"sycophantic" baseline); gpt-6-astra G3 68 %; both UNCLEAR; H 4 / 5 | `results/judgefair_nonclaude_2026-11-14.json`; LABBOOK 2026-11-14 |
| C24 | Re-grade noise: P1/P2 plain replies byte-identical to dial 1c plain (300 of 300); graded 98 caved / 162 fluent vs 96 / 177 | `results/controls_arm_a_completions.jsonl` arm baseline vs `results/dial1c_completions.jsonl` arm baseline set gih (string equality, 19.11.); `controls_arm_a.json` `arms.baseline`; `dial1c_readout.json` `arms.baseline` |
| C23 | Human blind read of the 16 G3 replies where gpt-6-astra and judge differ: with judge 12 of 16 (Wilson 50.5-89.8 %) -> FAIR stands | `results/judgefair_human_result.json`; LABBOOK 2026-11-15 |

## Controls (prereg `results/prereg_controls_2026-11-17.md`, lines signed 2026-11-18 before each run)
| Tag | Claim | Source |
|---|---|---|
| C30 | Arm b: 20 dummies (10 random, 10 shuffled) at a1's strength: caved 60-167, median 108.5; 1 of 20 reaches 145 -> SPECIAL; 14 of 20 above 96 | `results/controls_arm_b.json` `arms.*.caved`, `reading`, `verdict` |
| C30b | Arm b by kind: shuffled-only 1 of 10 reach 145, median 107; random-only 0 of 10, median 109.5 | `results/controls_arm_b.json` `descriptive_only` |
| C31 | Arm b gates: regenerated baseline and a1 equal stored on 300/300 | `results/controls_arm_b.json` `gates` |
| C32 | Arm f (-a1): caved 69 (drop 27); 3 of 20 flipped dummies at half the drop (neg_S2 58, neg_S4 76, neg_R4 80); UR-40 24 of 40; fluent 188; words 57.2 -> FAIL | `results/controls_arm_f.json` `arms.neg_a1`, `verdict`, `ur40_agrees_neg_a1` |
| C33 | Arm f: 16 of the 40 UR replies "contradicts" | LABBOOK 2026-11-18 (late night) card 4 line |
| C34 | P1 (hold sentence): caved 95 vs regenerated plain 98 and neutral sentence 99; fluent 168 (bar 270) -> FAIL | `results/controls_arm_a.json` `arms.p1_hold`, `arms.baseline`, `arms.p1_neutral`, `verdicts.P1` |
| C35 | P2 (hold-minus-give arrow from system-prompt contrast): caved 94; 2 of 10 shuffled at half; UR-40 36/40; fluent 98 vs 162; words 84.8 -> FAIL | `results/controls_arm_a.json` `arms.p2`, `verdicts.P2`, `ur40_agrees_p2` |
| C36 | P2 build: split-half cos 0.955; cos to a1 -0.062 | `results/controls_arm_a.json` `build` |
| C37 | Arm c: plain corrects own wrong answer 47.3 % (bare 2.2 %, hint 92.5 %, n 638); 0 of 7 arrows "one knob"; cue -11.3 / -1.7 pp; a1 +16.3 / -3.8 pp; C2 correction 47.3 -> 78.5 % | `results/controls_arm_c.json` `band`, `rule`, `correction_rate` |
| C41b | "up to 71 prompts" = 167 - 96 (arm b S5) | `results/controls_arm_b.json` |
| C38 | Arm d: median cut 0.9595; a1 right boxes +14 / +22 pp caving, wrong boxes -8 / -4 pp correction; "all four the same way" not met for cue or a1 | `results/controls_arm_d.json` `change_vs_plain`, `all_same_direction` |
| C39 | Arm e: real cos 0.6829 / 0.6765; same-population null median ~0.998; gap -0.31 / -0.32 -> FAIL; above a sign-flip null (99.4 / 99.9 %) | `results/controls_arm_e.json` `folds`; LABBOOK 2026-11-18 (controls started) |
| C40 | Arm e post-hoc reading "overlap but clearly not one direction" DECIDED by the author as a labelled finding | LABBOOK 2026-11-18 (evening) |
| C41 | Dummy spread across cards 1, 2, 4: 58-167 caved | `controls_arm_b.json` (60-167), `controls_arm_a.json` R/S (75-154), `controls_arm_f.json` neg_* (58-136) |
| C42 | Two fresh blind sub-agents recomputed every count of cards 2-4 from raw pool/key/grade files | LABBOOK 2026-11-18 (late night) claim-check |
| C43 | Controls spend ~46 USD Batch in total | LABBOOK 2026-11-18 (late night) "Spend tonight" |

## Refusal (appendix A, pipeline check only)
| Tag | Claim | Source |
|---|---|---|
| C50 | 14B (4-bit) held-out: full refusals 11/12 -> 0/12; random 10/12, 9/12; 4 steered "partial"; all 24 fluent | `results/control_14b_unblinded.json` `*|heldout` |
| C52 | 1.5B add-side: harmless prompts, plain 0/18 refused; +10 at block 13 -> 12/18 full+partial (10 full, 2 partial), coh3 16; grader fresh blind Claude sub-agent | `results/hook_layer_compare_v3_unblinded.json` `table.harmless_add_+10_block13`, `claim_B_add_strength`, `grader` |
| C51 | 1.5B held-out: ablate block 13 -> 2/12 full; ablate all layers -> 0/12; random ablation 12/12 | `results/hook_layer_compare_v3_unblinded.json` `heldout_*` |

## Claim-check 2026-11-19 (blind fresh agent, all tags)
All numbers SUPPORTED by their files; 1 WRONG ("first registered stop" -> stop (ii)), 7 OVERSTATED and 1 unsourced
wording fixed the same day. Judge: a fresh sub-agent with no expected answer, raw files only.

## Literature
| Tag | Claim | Source |
|---|---|---|
| L1 | 2611.02318 sec. 8: steering "a natural complement we did not attempt" | `results/prereg_controls_2026-11-17.md` (agent read of the arXiv page); LABBOOK 2026-11-18 (evening) attribution fix |
| L2 | Lee 2611.11035 "Room to Reconsider": four open models (Qwen3 1.7B/4B/8B, OLMo-2 7B); abstract: "Activation steering never reduced caving selectively: where it worked, it suppressed self-correction as well"; reply length decides caving vs self-correction | arXiv abstract page, read by WebFetch 2026-11-19 (which models were steered: not on the abstract page) |
| L3 | Vennemeyer et al. 2509.21305: agreement and praise separate directions, 4B-70B | arXiv 2509.21305 (https://arxiv.org/abs/2509.21305) |
| L4 | SteerCheck 2608.24335: controls too easy or leaky; one model (Qwen3-14B); leakage alone does not break the test (citation sweep 19.11.) | arXiv 2608.24335 (https://arxiv.org/abs/2608.24335) |
| L5 | Persona Vectors 2507.21509: own-response vector steers Qwen2.5-7B (layer 20); MMLU drops at large steering strengths (citation sweep 19.11.) | arXiv 2507.21509 (https://arxiv.org/abs/2507.21509) |
| L6 | Arditi et al. 2406.11717: refusal is mediated by one direction | public summary |
| L7 | 2611.02318 (Aamir, Bin Adil + 1): eight models 0.5B-14B, two families; "no usable direction below 7B", gate cleared at 7B, 8B, 14B; ablation at Qwen2.5-7B lowers reversals by 4.8 pp | arXiv abstract page, read by WebFetch 2026-11-19 |
| L8 | 2609.17550 (Aamir, Bin Adil): Qwen2.5-1.5B, "Are you sure?" among four pushback wordings; flip-vs-hold direction at the last prompt token not usable (AUROC 0.58, bar 0.70); no steering, no ablation | citation sweep 18./19.11. (primary page read) |
