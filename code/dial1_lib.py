"""Dial 1 shared pieces: model load, the ONE hook (add / ablate), batched generation, judge-free metrics.

Everything in the prereg (results/prereg_dial1_2026-10-25.md) section 0, 3, 4 that does not need a judge.
The hook sits on model.model.layers[LAYER]'s output (= hidden_states[LAYER+1]); one block, no all-layers hook.
"""
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent  # repo root (scripts live in code/)
DATA = HERE / "dial1_data"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 17
FS = (0.05, 0.10, 0.20, 0.35, 0.50)


def rows(name):
    return [json.loads(l) for l in open(DATA / name, encoding="utf-8")]


def load(model_id=MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda").eval()
    return tok, model


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


class Dial:
    """mode None = off; ('add', d, alpha) or ('ablate', d). One forward hook on the dial block."""

    def __init__(self, model):
        self.mode = None
        self.h = model.model.layers[LAYER].register_forward_hook(self.fn)

    def fn(self, _m, _inp, out):
        if self.mode is None:
            return None
        x = out[0] if isinstance(out, tuple) else out
        kind, d = self.mode[0], self.mode[1].to(x.device, x.dtype)
        x = x + self.mode[2] * d if kind == "add" else x - (x @ d).unsqueeze(-1) * d
        return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x

    def set(self, mode):
        self.mode = mode


def batches(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


@torch.no_grad()
def generate(tok, model, prompts, max_new, bs=16):
    outs = []
    for b in batches(prompts, bs):
        enc = tok([chat(tok, p) for p in b], return_tensors="pt", padding=True).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def letter_ids(tok):
    return {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFG"}


@torch.no_grad()
def p_syc(tok, model, items, bs=16):
    """P(sycophantic letter) at the answer position, normalised over that item's option letters."""
    lid, out = letter_ids(tok), []
    for b in batches(items, bs):
        enc = tok([chat(tok, it["prompt"]) + "(" for it in b], return_tensors="pt", padding=True).to("cuda")
        lg = model(**enc).logits[:, -1].float()
        for it, row in zip(b, lg):
            opts = [it["syc"]] + it["other"]
            p = torch.softmax(torch.stack([row[lid[c]] for c in opts]), 0)
            out.append(p[0].item())
    return out


@torch.no_grad()
def resid_at_letter(tok, model, items, letters, bs=8):
    """Residual (hidden_states[LAYER+1]) at the answer-letter token: prompt + '(' + letter, read at the letter."""
    lid, vecs = letter_ids(tok), []
    for b, ls in zip(batches(items, bs), batches(letters, bs)):
        ids = [tok(chat(tok, it["prompt"]) + "(", add_special_tokens=False).input_ids + [lid[l]] for it, l in zip(b, ls)]
        n = max(map(len, ids))
        inp = torch.tensor([[tok.pad_token_id] * (n - len(i)) + i for i in ids]).cuda()
        att = torch.tensor([[0] * (n - len(i)) + [1] * len(i) for i in ids]).cuda()
        hs = model(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states[LAYER + 1]
        vecs.append(hs[:, -1].float().cpu())
    return torch.cat(vecs)


@torch.no_grad()
def mean_norm(tok, model, items, bs=16):
    """Mean residual norm at the answer position (the strength unit)."""
    ns = []
    for b in batches(items, bs):
        enc = tok([chat(tok, it["prompt"]) + "(" for it in b], return_tensors="pt", padding=True).to("cuda")
        ns.append(model(**enc, output_hidden_states=True).hidden_states[LAYER + 1][:, -1].float().norm(dim=-1).cpu())
    return torch.cat(ns).mean().item()


def first_number(s):
    m = re.findall(r"-?\d+", s.replace(",", ""))
    return int(m[0]) if m else None


def gk_hit(out, answers):
    o = out.lower().strip().rstrip(".")
    return any(a == o or re.search(rf"\b{re.escape(a)}\b", o) for a in answers)


@torch.no_grad()
def kl_text(tok, model, dial, mode, texts, bs=10):
    """Mean per-token KL(steered || unsteered) along the UNSTEERED continuation (32-token prefix, 64 new)."""
    tot, n = 0.0, 0
    tok.padding_side = "left"
    for b in batches(texts, bs):
        pre = [tok.decode(tok(t, add_special_tokens=False).input_ids[:32]) for t in b]
        enc = tok(pre, return_tensors="pt", padding=True).to("cuda")
        dial.set(None)
        g = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        att = (g != tok.pad_token_id).long()
        att[:, :enc.input_ids.shape[1]] = enc.attention_mask
        lg0 = model(input_ids=g, attention_mask=att).logits[:, enc.input_ids.shape[1] - 1:-1].float()
        dial.set(mode)
        lg1 = model(input_ids=g, attention_mask=att).logits[:, enc.input_ids.shape[1] - 1:-1].float()
        dial.set(None)
        m = att[:, enc.input_ids.shape[1]:].bool()
        # kl_div(input=log q, target=log p) = KL(p||q): target = steered (lg1), input = unsteered (lg0)
        kl = F.kl_div(F.log_softmax(lg0, -1), F.log_softmax(lg1, -1), log_target=True, reduction="none").sum(-1)
        tot += (kl * m).sum().item()
        n += m.sum().item()
    return tot / max(n, 1)
