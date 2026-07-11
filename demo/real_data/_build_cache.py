"""Normalize cached raw datasets-server JSON pages into clean JSONL.

Reads the _raw_*.json pages (verbatim HF datasets-server /rows responses that
were fetched once at build time) and writes:
  openbookqa.jsonl  {"id","question","answer"}   (answer = correct choice text)
  ag_news.jsonl     {"id","text"}
"""
import json, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))

def load_pages(prefix):
    rows = []
    for fp in sorted(glob.glob(os.path.join(HERE, f"_raw_{prefix}_*.json"))):
        with open(fp, encoding="utf-8") as fh:
            txt = fh.read().strip()
        if not txt:
            continue
        obj = json.loads(txt)
        rows.extend(obj["rows"])
    return rows

def build_obqa():
    rows = load_pages("obqa")
    out = []
    for r in rows:
        d = r["row"]
        ch = d["choices"]
        idx = ch["label"].index(d["answerKey"])
        ans = ch["text"][idx]
        out.append({"id": f"obqa-{d['id']}", "question": d["question_stem"], "answer": ans})
    # de-dup by id, keep order
    seen, uniq = set(), []
    for x in out:
        if x["id"] in seen: continue
        seen.add(x["id"]); uniq.append(x)
    with open(os.path.join(HERE, "openbookqa.jsonl"), "w", encoding="utf-8") as fh:
        for x in uniq:
            fh.write(json.dumps(x) + "\n")
    return len(uniq)

def build_agnews():
    rows = load_pages("agnews")
    out = []
    for i, r in enumerate(rows):
        d = r["row"]
        text = d["text"].replace("\\\\", " ").replace("\\", " ").strip()
        out.append({"id": f"agnews-{i:04d}", "text": text})
    with open(os.path.join(HERE, "ag_news.jsonl"), "w", encoding="utf-8") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    return len(out)

if __name__ == "__main__":
    n1 = build_obqa()
    n2 = build_agnews()
    print(f"openbookqa.jsonl: {n1} items")
    print(f"ag_news.jsonl: {n2} docs")
