"""Entanglement, measured with POWER: does removing one clean direction disturb
another, and does it disturb *knowledge* at all?

Upgrade over entanglement.py (which used an 8-question quiz -> +-17pt coin-flip):
  * France quiz 8 -> 40 questions   (binomial CI shrinks ~ 1/sqrt(N))
  * a GENERAL-knowledge control quiz (facts unrelated to France) so we can tell
    "damaged France facts" from "damaged knowledge in general" (specificity)
  * Wilson 90% confidence intervals on every rate, so a change is either
    outside-noise (real) or inside-noise (not shown) -- with a number, not a vibe
  * one script runs BOTH models: 1.5B in fp16, 14B in 4-bit (bitsandbytes)
    via env vars, so the 1.5B baseline and the 14B run are directly comparable.

Run:
    set MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct  & python entanglement2.py   # fp16
    set MODEL_ID=Qwen/Qwen2.5-14B-Instruct & set LOAD_4BIT=1 & python entanglement2.py
"""

import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the validated instrument. These take (model, tok, ...) so they are
# model-agnostic; only refusal_direction.MODEL is 1.5B and we don't use it.
from refusal_direction import (
    REFUSED, ANSWERED, chat, last_token_resid, generate, refusal_rate,
)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

# A second CLEAN, LINEAR direction: "thinking about France" vs neutral topics.
FRANCE = [
    "Tell me about the city of Paris.",
    "What language do people speak in France?",
    "Describe the Eiffel Tower.",
    "What is a famous French dish?",
    "Who was Napoleon?",
    "Name a major river in France.",
    "What is the French flag like?",
    "Tell me about the French Riviera.",
    "Describe the Louvre museum.",
    "What is the Palace of Versailles?",
]
NEUTRAL = [
    "Tell me about baking sourdough bread.",
    "What do house cats like to eat?",
    "Describe how a bicycle works.",
    "What is a famous Italian dish?",
    "Who was Isaac Newton?",
    "Name a major river in Brazil.",
    "What is the Japanese flag like?",
    "Tell me about tropical beaches.",
    "Describe the game of chess.",
    "What is a rainbow?",
]

# 40 France facts with a checkable answer substring. Ablating REFUSAL must not
# dent this if refusal and France are independent; ablating FRANCE should.
FRANCE_QUIZ = [
    ("What is the capital of France?", "paris"),
    ("The Eiffel Tower is located in which city?", "paris"),
    ("What currency is used in France today?", "euro"),
    ("What is the official language of France?", "french"),
    ("On which continent is France located?", "europe"),
    ("What sea borders southern France?", "mediterranean"),
    ("What is the capital city of France, in one word?", "paris"),
    ("France is a country in which continent?", "europe"),
    ("Which famous tower in Paris was built for the 1889 World's Fair?", "eiffel"),
    ("What is the name of the famous art museum in Paris that holds the Mona Lisa?", "louvre"),
    ("Which river runs through Paris?", "seine"),
    ("What is the national day of France called, celebrated on July 14?", "bastille"),
    ("Who was the French emperor defeated at Waterloo?", "napoleon"),
    ("What is the tallest mountain in the French Alps?", "mont blanc"),
    ("What French palace was home to Louis XIV?", "versailles"),
    ("What is the famous cathedral in Paris that suffered a fire in 2019?", "notre"),
    ("What color is the left stripe of the French flag?", "blue"),
    ("What is a common French bread, long and thin?", "baguette"),
    ("What French region is famous for its sparkling wine?", "champagne"),
    ("What is the name of the cycling race held in France each summer?", "tour de france"),
    ("What is the largest city on the French Riviera?", "nice"),
    ("Which French city is famous for its film festival?", "cannes"),
    ("What snail dish is a French delicacy?", "escargot"),
    ("What is the French word for 'thank you'?", "merci"),
    ("What ocean lies to the west of France?", "atlantic"),
    ("Which country lies directly across the Channel from France?", "england"),
    ("What is the name of France's high-speed train?", "tgv"),
    ("What French pastry is a flaky crescent roll?", "croissant"),
    ("Who painted many water lilies and lived at Giverny in France?", "monet"),
    ("What is the second-largest city in France, a Mediterranean port?", "marseille"),
    ("What is the French region known for lavender fields?", "provence"),
    ("What is the name of the avenue in Paris ending at the Arc de Triomphe?", "champs"),
    ("What is the French national motto's first word (liberty)?", "liberty"),
    ("What mountain range separates France from Spain?", "pyrenees"),
    ("What is the French word for 'yes'?", "oui"),
    ("What famous French fashion designer created the 'little black dress'?", "chanel"),
    ("What is the northern French region famous for World War I battles?", "somme"),
    ("What type of government does France have (a republic)?", "republic"),
    ("What is the French Riviera also called (Cote d'...)?", "azur"),
    ("What white soft cheese, like Camembert, comes from France?", "brie"),
]

# 20 GENERAL facts, nothing to do with France. This is the specificity control:
# ablating the France direction should leave THESE alone if it is truly "topic".
GENERAL_QUIZ = [
    ("What is the capital of Japan?", "tokyo"),
    ("What planet is known as the Red Planet?", "mars"),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide"),
    ("How many continents are there on Earth?", "seven"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What is the chemical symbol for gold?", "au"),
    ("Who wrote the play 'Romeo and Juliet'?", "shakespeare"),
    ("What is the hardest natural substance on Earth?", "diamond"),
    ("What is the capital of Australia?", "canberra"),
    ("How many legs does a spider have?", "eight"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the largest planet in our solar system?", "jupiter"),
    ("What metal is liquid at room temperature?", "mercury"),
    ("Who developed the theory of general relativity?", "einstein"),
    ("What is the smallest prime number?", "2"),
    ("What is the tallest animal in the world?", "giraffe"),
    ("What is the main gas in Earth's atmosphere?", "nitrogen"),
    ("What is the capital of Egypt?", "cairo"),
    ("How many sides does a hexagon have?", "six"),
    ("What organ pumps blood through the human body?", "heart"),
]


def wilson(k, n, z=1.645):
    """Wilson score interval (default 90%). Returns (lo, hi) as fractions."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def direction(model, tok, pos, neg, layer):
    d = last_token_resid(model, tok, pos, layer) - last_token_resid(model, tok, neg, layer)
    return d / d.norm()


def quiz(model, tok, questions):
    """Return (hits, n) on a substring-match quiz."""
    hits = 0
    for q, ans in questions:
        out = generate(model, tok, q, max_new=24).lower()
        hits += ans in out
    return hits, len(questions)


class Ablate:
    """Project direction d OUT of the residual stream at every layer: x-(x.d)d."""

    def __init__(self, model, d):
        self.handles = []
        d = d.to(model.device, model.dtype)
        for layer in model.model.layers:
            def hook(_m, _inp, output, d=d):
                x = output[0] if isinstance(output, tuple) else output
                proj = (x @ d).unsqueeze(-1) * d
                x = x - proj
                return (x,) + output[1:] if isinstance(output, tuple) else x
            self.handles.append(layer.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def load():
    print(f"Loading {MODEL_ID}  (4bit={LOAD_4BIT}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = {}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kw["device_map"] = "cuda"
    else:
        kw["torch_dtype"] = torch.float16
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    return model, tok


@torch.no_grad()
def main():
    model, tok = load()
    layer = len(model.model.layers) // 2
    print(f"{len(model.model.layers)} layers; directions at layer {layer}\n", flush=True)

    d_refusal = direction(model, tok, REFUSED, ANSWERED, layer)
    d_france = direction(model, tok, FRANCE, NEUTRAL, layer)

    # Reference points so cosine is readable.
    g = torch.Generator(device="cpu").manual_seed(0)
    rand = torch.randn(d_refusal.shape, generator=g).to(model.device, model.dtype)
    rand = rand / rand.norm()
    half = len(REFUSED) // 2
    d_ref_a = direction(model, tok, REFUSED[:half], ANSWERED[:half], layer)
    d_ref_b = direction(model, tok, REFUSED[half:], ANSWERED[half:], layer)

    def cos(a, b):
        return float((a.float() @ b.float()) / (a.float().norm() * b.float().norm()))

    geom = {
        "refusal_vs_france": cos(d_refusal, d_france),
        "refusal_vs_random_CONTROL": cos(d_refusal, rand),
        "refusal_selfconsistency_halfA_vs_halfB": cos(d_ref_a, d_ref_b),
    }
    print("GEOMETRIC overlap (cosine):")
    print(f"  refusal vs France          : {geom['refusal_vs_france']:+.3f}")
    print(f"  refusal vs RANDOM (control): {geom['refusal_vs_random_CONTROL']:+.3f}")
    print(f"  refusal A vs B (same thing): {geom['refusal_selfconsistency_halfA_vs_halfB']:+.3f}\n", flush=True)

    def measure(tag):
        rr = refusal_rate(model, tok, REFUSED)
        fh, fn = quiz(model, tok, FRANCE_QUIZ)
        gh, gn = quiz(model, tok, GENERAL_QUIZ)
        rec = {
            "refusal_rate": rr,
            "france_quiz": {"hits": fh, "n": fn, "acc": fh / fn,
                            "ci90": wilson(fh, fn)},
            "general_quiz": {"hits": gh, "n": gn, "acc": gh / gn,
                             "ci90": wilson(gh, gn)},
        }
        fl, fhi = rec["france_quiz"]["ci90"]
        gl, ghi = rec["general_quiz"]["ci90"]
        print(f"{tag:16s} refusal {rr:5.0%} | France {fh}/{fn}={fh/fn:.0%} "
              f"[{fl:.0%},{fhi:.0%}] | General {gh}/{gn}={gh/gn:.0%} "
              f"[{gl:.0%},{ghi:.0%}]", flush=True)
        return rec

    print("BASELINE + two ablations (90% Wilson CIs in brackets):")
    base = measure("baseline")

    ab = Ablate(model, d_refusal)
    abl_ref = measure("ablate-refusal")
    ab.remove()

    ab = Ablate(model, d_france)
    abl_fra = measure("ablate-france")
    ab.remove()

    results = {
        "model": MODEL_ID, "load_4bit": LOAD_4BIT, "layer": layer,
        "geometric_cosine": geom,
        "arms": {"baseline": base, "ablate_refusal": abl_ref, "ablate_france": abl_fra},
    }
    tag = MODEL_ID.split("/")[-1]
    path = OUT / f"entanglement2_{tag}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nNumbers -> {path}", flush=True)


if __name__ == "__main__":
    main()
