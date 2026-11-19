"""Dial 1b: finish the blind grading via the Gemini Batch API (resume after the 402 credit stop, 09.11.).

Same model, same prompt text (dial1b_judge.batch_text), same chunks of 20 in pool order, same parser,
same grades file as `dial1b_judge.py grade`. Only the transport differs (Batch instead of one call each).

    python dial1b_judge_batchapi.py submit N   # submit the first N ungraded chunks as one batch job
    python dial1b_judge_batchapi.py poll       # wait for the open job, write grades, print cost
Cost is counted at the STANDARD price (2 / 12 USD per 1M in / out tokens), an upper bound for Batch.
"""
import json
import sys
import time

import dial1b_judge as B

L = B.J.llm_api
BASE = "https://generativelanguage.googleapis.com/v1beta/"
JOBS = B.RES / "dial1b_batch_jobs.json"
PRICE_IN, PRICE_OUT = 2.0, 12.0


def hdr():
    return {"x-goog-api-key": L.read_key("gemini"), "Content-Type": "application/json; charset=utf-8"}


def todo_chunks():
    pool = [json.loads(l) for l in open(B.POOL, encoding="utf-8") if l.strip()]
    res = json.loads(B.GR.read_text())["grades"]
    todo = [p for p in pool if p["pid"] not in res]
    return [todo[i:i + 20] for i in range(0, len(todo), 20)]


def jobs():
    return json.loads(JOBS.read_text()) if JOBS.exists() else []


def submit(n):
    it = B.items()
    chunks = todo_chunks()[:n]
    reqs = [{"request": {"contents": [{"role": "user", "parts": [{"text": B.batch_text(it, c)}]}]},
             "metadata": {"key": ",".join(p["pid"] for p in c)}} for c in chunks]
    body = {"batch": {"display_name": f"dial1b-grade-{int(time.time())}",
                      "input_config": {"requests": {"requests": reqs}}}}
    r = L._http(BASE + f"models/{L.MODELS['gemini']}:batchGenerateContent", hdr(), json.dumps(body).encode("utf-8"))
    js = jobs() + [dict(name=r["name"], chunks=len(chunks), items=sum(map(len, chunks)), model=L.MODELS["gemini"],
                        submitted=time.strftime("%Y-%m-%d %H:%M:%S"), state="submitted")]
    JOBS.write_text(json.dumps(js, indent=1))
    print("submitted", r["name"], len(chunks), "chunks", sum(map(len, chunks)), "items")


def poll():
    js = jobs()
    job = next(j for j in js if j["state"] == "submitted")
    while True:
        r = L._http(BASE + job["name"], hdr())
        st = r.get("metadata", {}).get("state", "?")
        if r.get("done"):
            break
        print(time.strftime("%H:%M:%S"), st, flush=True)
        time.sleep(60)
    it = B.items()
    pool = {p["pid"]: p for p in map(json.loads, open(B.POOL, encoding="utf-8"))}
    gr = json.loads(B.GR.read_text())
    res = gr["grades"]
    tin = tout = ok = bad = 0
    resp = r.get("response", {}).get("inlinedResponses", {}).get("inlinedResponses", [])
    for x in resp:
        chunk = [pool[pid] for pid in x["metadata"]["key"].split(",")]
        g = x.get("response") or {}
        u = g.get("usageMetadata") or {}
        tin += u.get("promptTokenCount", 0)
        tout += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        try:
            parts = g["candidates"][0]["content"]["parts"]
            raw = "\n".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
            res.update({p["pid"]: dict(grade=a, coherence=c) for p, (a, c) in zip(chunk, B.parse_list(it, chunk, raw))})
            ok += 1
        except Exception as e:  # stays ungraded; the next submit picks it up again with the same prompt
            bad += 1
            print("unparsed chunk", chunk[0]["pid"], type(e).__name__, str(x.get("error", ""))[:120])
    B.GR.write_text(json.dumps(dict(gr, grades=res)))
    usd = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    job.update(state=r.get("metadata", {}).get("state", "done"), ok_chunks=ok, bad_chunks=bad, tokens_in=tin,
               tokens_out=tout, usd_standard_price=round(usd, 4), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    JOBS.write_text(json.dumps(js, indent=1))
    print(f"done: {ok} chunks ok, {bad} unparsed, graded total {len(res)}; tokens in {tin} out {tout}; "
          f"<= {usd:.3f} USD at standard price (all jobs: {sum(j.get('usd_standard_price', 0) for j in js):.3f})")


if __name__ == "__main__":
    submit(int(sys.argv[2])) if sys.argv[1] == "submit" else poll()
