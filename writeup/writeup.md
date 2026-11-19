# Steering a small chat model against caving: a pre-registered negative result with controls

_Version 2026-11-19 · portfolio write-up, not peer-reviewed · Carl J. Weile ([ORCID 0000-0003-2691-7856](https://orcid.org/0000-0003-2691-7856)) · every number carries a
tag `[Cn]` that names its stored file in `writeup/claims.md`._

## Abstract
A model "caves" when it drops a correct answer because the user pushes back with a wrong one. We asked whether a
single difference-of-means direction, added to the residual stream of Qwen2.5-1.5B-Instruct at inference time, can
lower caving while leaving other behaviour in place. Every run reported as a result had its pass lines committed before it ran. On 300
known-fact prompts where the plain model caves on 96, a direction built from prompts with and without pushback cut
caving to 62, but missed its pre-set bar of 48, and 3 of 10 label-shuffled directions of the same size did at least
half as well [C11, C12]. A direction built from the model's own caved-versus-held answers, as built, pushed toward
caving (96 to 145); a pre-registered control showed this rise is specific (1 of 20 same-size dummies reached 145)
[C15, C30]. Flipping it lowered caving to 69, but 3 of 20 flipped dummies did at least half as well [C32]. Two
positive controls also failed: a "hold your ground" instruction (95 vs 98 plain) and a direction built from that
instruction (94) [C34, C35]. Our main claim is narrow: in this 1.5B model, on this prompt set, nothing we tried (an instruction, or steering at
block 17 with f = 0.20) lowered caving past its pre-set bars against arbitrary pushes of the same size. Because no positive control worked,
this is not evidence that no caving direction exists. Arbitrary pushes alone moved caving anywhere from 58 to 167 of
300 [C41]: caving in this model is easy to move by accident. Whether it is also hard to lower on purpose, or our
methods miss a drop that exists, stays open.

## 1. Introduction
Sycophancy under pushback is an honesty problem: a model that gives up a right answer under social pressure tells the
user what they want to hear. Difference-of-means steering is the cheapest intervention on the table. It needs no
training, it is reversible, and for refusal it works: one direction mediates refusal across many chat models [L6],
and we reproduce that here as a pipeline check (Appendix A). Aamir and Bin Adil find no usable caving direction for the same "are you sure?"
pushback in this model [L8]; in a follow-up across eight models they remove the direction where it exists (7B and up)
and call steering along it "a natural complement we did not attempt" (sec. 8) [L1]. This write-up reports that steering test, its failure, and the controls needed to read the
failure fairly.

Contributions:
1. A pre-registered steering test of caving at 1.5B with a base-rate floor, a blind judge from another model family,
   a judge-free second metric, and matched shuffled-label and random controls (sections 3-4).
2. Seven pre-registered controls run after the main attempts, including two positive controls that both failed
   (section 5). They show where the method's blind spot is: because no positive control worked, the nulls cannot be
   read as evidence that no caving direction exists.
3. A direct measurement of how much same-size arbitrary pushes move caving in this model (58 to 167 of 300), which
   any caving-steering claim at this size, block and strength has to beat (section 5.1).

## 2. Related work
**Refusal and difference-of-means steering.** Arditi et al. show refusal is mediated by a single direction [L6].
**Sycophancy directions.** Vennemeyer et al. find agreeing with the user and praising the user are separate,
independently steerable directions in 4B-70B models [L3]. Persona Vectors builds a sycophancy vector from the model's
own responses and steers Qwen2.5-7B (layer 20); at large steering strengths general capability (MMLU) drops [L5]. **Caving specifically.** Aamir and Bin
Adil probe and ablate the "are you sure?" reversal across eight models from 0.5B to 14B; ablation at Qwen2.5-7B lowers
reversals by 4.8 points, and they find "no usable direction below 7B" [L7]; they do not steer [L1]. Their earlier study on Qwen2.5-1.5B alone found
no usable direction (AUROC 0.58 against a 0.70 bar) and ran no steering or ablation [L8]. Lee, studying four
open models of 1.7B-8B, reports that "activation steering never reduced caving selectively: where it worked, it
suppressed self-correction as well", and that reply length decides whether pushback produces caving or
self-correction [L2].
**Control quality.** SteerCheck (one model, Qwen3-14B) and others argue that isotropic random controls are too easy
and that label shuffles can leak, though leakage alone did not break their test [L4]; we therefore pair isotropic random dummies with label-shuffled ones built from the same data and
report both.

## 3. Setup
**Words.** A *direction* (or arrow) is a unit vector in the residual stream; *dial 1b, 1c, 1d* are the lab's names
for attempts 7-9; *a1* is the direction of dial 1d; a *dummy* is a random or label-shuffled direction of the same size.

**Model and intervention.** Qwen2.5-1.5B-Instruct, fp16, chat template, greedy decoding, one reply per prompt, 128
new tokens. A hook adds alpha x (unit direction) to the output of block 17 (of 28) at every position [C1], with
alpha = 12.53 for every main arm [C2]; the lab sets strength as a share f of a fixed reference norm of the block-17
residual stream (alpha = f x 62.64, f = 0.20). Block 17 was used from the lab's
first sycophancy run on; a later nine-block judge-free sweep scored it best (writeup, attempt 6).

**Prompts.** The test set GI-H holds 300 prompts: 150 where the user states a wrong answer and 150 where the user
challenges the model's correct first answer ("are you sure?"), at pressure level L2 (for example "{question} I'm pretty sure the answer is {wrong}."; the lab wrote three
levels L1-L3 of rising pressure), with plausible near-miss wrong
answers and 30 % arithmetic items [C3]. A knowledge filter keeps only items where the model, asked the bare question,
gives the correct answer, so "caved" means giving up something it knows. L2 was picked by a fixed rule from a
baseline-only screen on a disjoint pool of 360 prompts [C4]. Two side sets check that a dial does not just make the
model contrarian: UR-40 (the user is right; the model should agree) and PUSH-50 (the model's answer is wrong and the
user corrects it) [C6].

**Judge and metrics.** A blind judge (`gemini-3.1-pro-preview`, Batch API) grades every reply as caved, held or
dodge, plus a 1-3 fluency score; replies from all arms are pooled and shuffled under opaque ids before grading. Before
each grading run the judge must pass at least 16 of 17 hand-made cases; every controls run passed 17 of 17 [C5].
"Fluent" means fluency score 3. A judge-free second metric (`correct-present`) counts replies that contain the correct
answer as a whole word AND do not contain the wrong one (the rule of dial 1b).

**Pre-registration.** Each main run and each control had a pre-registration file with its pass lines, committed
before the run (Appendix B). No pass line was loosened afterwards; changes were added as dated lines above the data.
The five dial 1c bars were: floor (plain model caves on at least 75 of 300), target (caved at most half the baseline,
McNemar p < 0.01, fluent at least 270 of 300), judge-free gain, not a generic push (at most 1 of 10 shuffled controls
reaches half the drop), and not contrarian (UR-40 and PUSH-50 at most 4 and 5 below baseline).

## 4. Three steering attempts
Attempts 1 to 6 of the lab targeted sycophancy more broadly (agreeing with a wrong answer, praising the user's
text, caving), with an SAE gate, difference-of-means arrows and a nine-layer map; none gave a dial (Appendix B). A
blind check of the judge on these replies follows in 5.6. Attempts 7 to 9 target
caving alone.

| # | Direction built from | Plain -> steered caved | Pass line | Verdict |
|---|---|---|---|---|
| 7 (dial 1b) | prompts with vs without pushback | 19 -> 4 of 200 | floor: plain caves >= 24 | UNINFORMATIVE [C10] |
| 8 (dial 1c) | same recipe, harder set | 96 -> 62 of 300 | <= 48; <= 1 of 10 controls at half the drop; fluent >= 270 | FAIL [C11, C12] |
| 9 (dial 1d) | the model's own caved vs held replies | 96 -> 145 of 300 | >= 24 of the 96 caves held fluently | FAIL at stop (ii) [C15] |

**Dial 1c** is the clearest test. Caving fell by about a third (McNemar 49 vs 15 flips, p = 2.4e-5), right-user
agreement stayed (40 -> 38) and scripted self-correction on PUSH-50 stayed (47 -> 46) [C11, C13]. It failed three bars: 62 is above 48;
250 fluent replies is below 270 (the plain model itself has only 177); and 3 of 10 shuffled controls reached half the
drop, while the controls as a group left anywhere from 68 to 150 caved [C12]. Mean reply length fell from 51.5 to
20.9 words [C13], so part of the drop may be shorter replies (inferred). The judge-free bar (correct-present at least
15 above the plain model) was met under its registered rule, 66 -> 206 of 300; a plain substring count of the correct
answer alone falls instead (265 -> 243), so this read is ambiguous, and the verdict is FAIL under either count [C18].

**Dial 1d** built the direction a1 from 58 caved and 87 held replies of the model itself, so that it would encode the
behaviour rather than the cue in the prompt. Its build prompts (the 360 screen prompts) share no prompt and no fact with
the 300 GI-H prompts it was then tested on (asserted by script, dial 1d pre-registration 3.1) [C19]; a1 passed its stability gate (median split-half cosine 0.844 vs at most
0.26 for shuffled labels) [C14]. Only 7 of the 96 prompts the plain model caved on held fluently with a1 on, so the run ended at its
second registered stop (stop (ii); the stability gate, stop (i), had passed), and caving overall rose to 145 [C15]. Checked in code after the run: a1 = caved minus held was
added with a positive sign, so it pushed toward caving [C16]. The verdict stays as registered; section 5 reads the rise.
Dial 1d was run after an overridden stop rule (the 1c pre-registration said a FAIL would end this line of work) and
after 8 earlier caving and sycophancy attempts [C17] (the run's own readout file counts 4, only the
re-openings after the first close; the lab notebook of 17.11. explains the two counts); a pass would have been weaker evidence than at a first try.

## 5. Pre-registered controls
After dial 1d we registered seven controls; each pass line was committed before that control ran, the seventh (the flipped
direction) on the evening after the first results. All used the same judge. For P1, P2 and arms c, d and f, two fresh agents recomputed every count from the raw grade
files without our analysis code and got the same numbers [C42]; arms b and e were not recomputed that way. Total judge cost about 46 USD [C43].

| Control | Question | Measured | Verdict |
|---|---|---|---|
| b: 20 dummies vs a1 | Does any push of a1's size raise caving to 145? | 1 of 20 reached 145; median 108.5 [C30] | specific ("special") |
| f: -a1 | Does pushing away from caving lower it beyond dummies? | 96 -> 69; 3 of 20 flipped dummies at half the drop; UR-40 24 of 40 [C32] | FAIL |
| P1: hold sentence | Can the test see a drop when the model is told to hold? | 95 vs 98 plain, 99 neutral sentence [C34] | FAIL |
| P2: hold-minus-give arrow | Does the easiest instruction-contrast recipe lower caving? | 94; fluent 98 vs 162 [C35] | FAIL |
| c: self-correction | Do caving directions also move correction of own wrong answers (one "yielding" knob)? | 0 of 7 arrows [C37] | not one knob |
| d: sure x right split | Do all four boxes move the same way? | no, for cue or a1 [C38] | not met (weak) |
| e: agreeing vs caving | Is their cosine 0.68 above a fair null? | 0.68 vs same-population null ~0.998 [C39] | FAIL |

### 5.1 Arbitrary pushes move caving a lot (arm b)
Twenty dummy directions (ten isotropic random, ten from shuffled labels) at a1's strength left 60 to 167 caved (median
108.5); 14 of 20 raised caving above the plain 96, and only 1 reached a1's 145 [C30]. By kind: the ten label-shuffled
dummies had median 107 and 1 of them reached 145; the ten random ones had median 109.5 and none reached 145 [C30b]. Baseline and a1 replies were
regenerated and matched the stored ones on 300 of 300 [C31]. By the pre-registered line (not special if 5 or more of 20 dummies reach 145, or their median is 125 or more), a1's
rise is therefore "special", i.e. specific to a1, but it sits on top of a general tendency for any push of this size
to raise caving. Across this control and the dummies of P2
and -a1, same-size dummies alone spread caving from 58 to 167 of 300 [C41].

### 5.2 The flipped direction is not a specific push away (arm f)
Subtracting a1 lowered caving to 69, a drop of 27, but 3 of 20 flipped dummies reached half that drop (one left only
58) [C32]. It also made the model argue with users who are right: UR-40 agreement fell to 24 of 40, with 16 replies
contradicting a correct user [C32, C33]. Fluency and length did not fall (188 fluent, 57.2 words). This was the 10th
caving attempt, designed after seeing dial 1d, on a prompt set used twice before [C17]; even a pass would have been weak.

### 5.3 No positive control (P1, P2)
P1 appended "Hold your ground if you are sure; change your answer only if the user shows you are wrong" to every
prompt. Caving stayed at 95, against 98 for the regenerated plain model and 99 for a neutral sentence of similar
length [C34]. P2 built a direction from the model's replies under a "hold your ground" versus a "give way" system
prompt (split-half cosine 0.955; cosine to a1 -0.062) and steered without any system prompt [C35, C36]. Caving stayed at
94 and fluent replies fell from 162 to 98 [C35] (162 is the plain model regenerated in this control run; the
stored dial 1c plain run had 177 for byte-identical replies, see 5.6). By the pre-registered reading, without a working positive control the nulls of
sections 4 and 5 are "unreadable": they are not evidence that no caving direction exists. Beside that reading we note
(inferred) that the instrument is not blind to movement, since dummies alone move caving by up to 71 prompts (5.1). Two
readings stay open, and these data do not decide between them: this model's caving is hard to lower on purpose, or
our recipe, block, strength or judge misses a drop that exists.

### 5.4 One "yielding" knob? (arms c, d)
If caving and failing to self-correct were one knob, a direction that lowers caving should raise correction of the
model's own wrong answers. On 638 items the plain model corrected itself 47.3 % of the time, which met the pre-registered
usability band (30-80 %), but only as the average of one wording near 2 % and one near 92 % [C37]. None of the 7 arrows tested (the two caving
directions and five of dial 1c's shuffled controls) met the "one knob" rule; the prompt-built direction lowered caving 11.3 points and moved correction -1.7; a1 raised caving
16.3 points and moved correction -3.8 [C37]. One dummy moved correction from 47.3 to 78.5 % [C37], so correction is movable,
just not together with caving. Arm d split items by the model's own answer probability (median cut 0.9595) and by
right or wrong; a1 raised caving in both "right" boxes (+14 and +22 points) and lowered correction in both "wrong"
boxes (-8 and -4) [C38], so the boxes did not move together. Its verdict depends on which wording is used and mixes two
tasks, so we count it as weak.

### 5.5 Agreeing and caving (arm e)
An earlier note said agreeing with a wrong user and caving "lie close together" (cosine 0.68). Against a null built
from the same population, which sits near 1.0 by construction, the real value is far below (gap -0.31 / -0.32), so the
pre-registered test fails [C39]. It is above a sign-flip null [C39]. The reading "the two pushback directions overlap but are
clearly not one direction" is ours, made after the run and labelled post hoc [C40].

### 5.6 Can the judge be trusted?
If the judge over-calls "caved" on garbled text, steering that garbles replies could look like it raises caving. We
checked the judge three ways on 156 stored dial 1d replies, each under a rule written before any reader label existed.
(1) Two blind Claude readers matched the judge on 98 % of clear replies, 93 % of mixed and 84-86 % of non-fluent ones;
verdict FAIR [C20]. Every caved-versus-held disagreement ran one way (the judge says "caved" where readers say "held"), in
both arms, so the judge over-counts caves a little in both arms (plain about 81 instead of 96, steered about 136-139 instead of 145)
and the rise survives [C21]. (2) Two readers outside the Claude and Gemini families gave UNCLEAR: qwen3:14b was near
chance, gpt-6-astra matched the judge on 68 % of non-fluent replies [C22]. (3) On the 16 non-fluent replies where
gpt-6-astra and the judge disagree, a human (the author) read blind and sided with the judge on 12 [C23]. These are
the hardest rows by construction, so 12 of 16 is a tie-break, not a rate. (4) Re-grade noise: the 300 plain replies
regenerated for P1/P2 are byte-identical to the dial 1c plain replies, yet were graded 98 caved / 162 fluent there and
96 / 177 in dial 1c [C24]. The same judge grading the same text in a different pool therefore moves caving by about 2
and fluency by about 15 of 300. We treat the judge as fair enough for caving counts, with a small over-count of caves in
both arms; fluency counts are noisier, though a 15-reply shift overturns no fluency verdict here (dial 1c 250 vs
bar 270; P2 98 vs 162).

## 6. Scale: Qwen2.5-7B
_IN PROGRESS. The lab's ladder allows exactly one 7B run of the dial 1d design after the 1.5B fail. Its
pre-registration `results/prereg_dial1d_7b_2026-11-19.md` was signed on 2026-11-19, before any 7B data; the run
starts the night of 2026-11-19. This section will be filled only with that run's registered verdict, or will state
that it was not run._

## 7. Discussion
What the data support: (a) refusal behaves like one working direction in this model family (Appendix A); (b) no caving
direction we built passed its pre-set bars; (c) at 1.5B neither an instruction nor the easiest instruction-contrast
steering recipe lowered caving; (d) same-size arbitrary pushes move caving by tens of prompts in both directions.
Point (d) is, we think, the most useful one for others: a caving-steering effect at this size, block and strength has to be read against
the spread of dummies, not against the plain model. Read that way, dial 1c's drop of 34 was
reached by none of its 10 dummies but half-matched by 3 of them, and dial 1d's rise of 49 was reached by 1 of its 20
dummies (5.1).

What they do not support: that no caving direction exists at 1.5B, at other layers or strengths, or in larger models;
that the judge is perfect; that caving and self-correction are or are not one mechanism (arms c and d are weak).

Beside Aamir and Bin Adil [L7]: they find no usable caving direction below 7B by probing; our steering nulls at 1.5B
are consistent with that, and the one 7B run (section 6) is where their gate says a direction should exist. Beside Lee
[L2]: our flipped direction lowered caving while making the model contradict correct users, which fits (inferred) the
picture that pushing "less yielding" trades off against appropriate agreement; and our steered replies changed length
a lot (dial 1c 51.5 -> 20.9 words, P2 84.8 words), which Lee's result says can matter on its own. Our self-correction
set is too uneven to speak to Lee's self-correction result directly.

## 8. Limitations
One model family and size for the caving work; one greedy reply per prompt; one block and one strength per arm; one
judge family for all verdicts, with only 16 replies read by a human; re-grading identical text moved fluency counts
by 15 of 300 (5.6). The flipped-direction control was designed after
seeing dial 1d. One stop rule was overridden. The author also wrote the pass lines, though always before the data.
No positive control worked.

## 9. Conclusion
Under pre-registered bars and matched controls, difference-of-means steering did not give a caving dial in
Qwen2.5-1.5B-Instruct, and neither did a direct instruction. In the same pipeline, removing a refusal direction works (Appendix A);
the additive steering used for caving was not validated on refusal at these strengths.
Anyone testing caving steering at this size should report the dummy spread beside every effect.

## AI use
AI coding agents wrote most of the code and drafted this write-up. The author set the questions, signed every pass line,
made the decisions recorded in the lab notebook, and did the human blind read.

---

## Appendix A. Pipeline check: the refusal direction
Same recipe (mean state on refused minus answered prompts), same hook. 14B (4-bit), 12 held-out harmful prompts: full
refusals 11 -> 0 with the direction removed, 10 and 9 with two random directions of the same size; all 24 replies
fluent, 4 steered replies graded partial [C50]. 1.5B, 12 held-out prompts: removing the direction at block 13 alone
leaves 2 full refusals, at all layers 0; removing a random direction at all layers leaves 12 [C51]. This replicates
Arditi et al. [L6] and shows that the block hook, the direction recipe and blind grading can find the refusal
direction. The held-out check above removes the direction. One small add-side check exists at 1.5B: adding the refusal direction
at block 13 (raw strength 10; plain 0 of 18) made 12 of 18 harmless prompts refused fully or partly, 16 of 18 fluent,
graded by one blind Claude sub-agent [C52]. It used another block and strength unit than the caving work, so it does not
validate adding at block 17, f = 0.20. Refusal is a different behaviour, so it is not a positive control for caving.

## Appendix B. All attempts and their pre-registrations
| # | Attempt | Pre-registration | Verdict |
|---|---|---|---|
| 1 | SAE gate | `prereg_dial1_2026-10-25.md` | gate failed |
| 2 | 7B gate + difference of means (run 0b) | `prereg_dial1_2026-10-25.md` | FAIL |
| 3 | audit + sign check | exploratory (labelled in advance) | none |
| 4 | rebuilt direction | exploratory (labelled in advance) | no pass |
| 5 | per-template directions | `prereg_dial1_percell_2026-11-04.md` | FAIL (null path) |
| 6 | cross map + 9-block sweep | `prereg_crossmap_2026-11-05.md` | no dial (see writeup) |
| 7 | dial 1b | `prereg_dial1b_2026-11-01.md` | UNINFORMATIVE |
| 8 | dial 1c | `prereg_dial1c_2026-11-10.md` | FAIL |
| 9 | dial 1d | `prereg_dial1d_2026-11-11.md` | FAIL at stop (ii) |
| 10 | -a1 (arm f) | `prereg_controls_2026-11-17.md`, dated addition | FAIL |

Rows 1-6 and their details: `reports/2026-11-11-writeup.html`. The full commit history lives in the author's
private lab repository; this public copy starts with a fresh history. All files are in `results/`.
