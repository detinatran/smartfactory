"""Re-run the LLM4IAS (arXiv:2409.18009) Inspection Station evaluation with a local model.

Inputs come from the authors' repo (github.com/YuchenXia/LLM4IAS):
  prompt_example.txt  - full agent prompt for the Inspection Station
  evaluation_data.xlsx - per-model test points (event log, correct command, model output, score)
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import openpyxl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent


def load_template():
    text = (HERE / "prompt_example.txt").read_text(encoding="utf-8")
    # The file ends with "Input:\n<example log>\n\nOutput:"; swap that log for each test point.
    head, _ = text.rsplit("Input:\n", 1)
    return head + "Input:\n<<LOG>>\n\nOutput:\n"


def load_rows(sheet, station):
    wb = openpyxl.load_workbook(HERE / "eval.xlsx", read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    header = rows[0]
    out = []
    for r in rows[1:]:
        if not r or r[0] != station:
            continue
        d = dict(zip(header, r))
        out.append({
            "nr": d["test point number"],
            "type": d["type"],
            "log": d["content"].strip(),
            "correct": d["correct command"],
            "author_command": d["command"],
            "author_score": d["Column1"],
        })
    return out


def norm(cmd):
    if cmd is None:
        return ""
    c = str(cmd).strip().rstrip(";.").replace('"', "'")
    return re.sub(r"\s+", "", c)


def parse_command(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0)).get("command", ""), m.group(0)
        except json.JSONDecodeError:
            pass
    m = re.search(r'"command"\s*:\s*"(.*?)"\s*[,}]', text, re.S)
    return (m.group(1) if m else ""), text


# OpenAI-compatible chat endpoints: model prefix -> (URL, env var holding the key)
PROVIDERS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
}


def api_chat(provider, model, prompt, retries=6):
    url, key_env = PROVIDERS[provider]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 256, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {os.environ[key_env]}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 30)
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)


def make_generator(model_name):
    provider, _, api_model = model_name.partition(":")
    if provider in PROVIDERS and api_model:
        return lambda prompt: api_chat(provider, api_model, prompt)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def generate(prompt):
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=256, do_sample=False)
        return tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--author-sheet", default="Qwen2-7B-Instruct_1")
    ap.add_argument("--station", default="Inspection Station")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    template = load_template()
    rows = load_rows(args.author_sheet, args.station)
    gpt4o = {r["nr"]: r for r in load_rows("gpt-4o", args.station)}

    generate = make_generator(args.model)

    results = []
    for r in rows:
        text = generate(template.replace("<<LOG>>", r["log"]))
        cmd, raw = parse_command(text)
        r.update({
            "ours_raw": raw,
            "ours_command": cmd,
            "ours_exact": norm(cmd) == norm(r["correct"]),
            "author_exact": norm(r["author_command"]) == norm(r["correct"]),
            "gpt4o_exact": norm(gpt4o[r["nr"]]["author_command"]) == norm(r["correct"]),
        })
        results.append(r)
        print(f"#{r['nr']:>2} {r['type'][:10]:<10} ours={r['ours_exact']!s:<5} author={r['author_exact']!s:<5} "
              f"| correct={r['correct']} | ours={cmd}", flush=True)

    def rate(key, sub):
        return sum(x[key] for x in sub) / len(sub) if sub else float("nan")

    summary = {}
    for name, sub in [("all", results),
                      ("SOP", [x for x in results if x["type"] == "SOP"]),
                      ("Unexpected", [x for x in results if x["type"] != "SOP"])]:
        summary[name] = {
            "n": len(sub),
            "ours_exact": rate("ours_exact", sub),
            "author_exact": rate("author_exact", sub),
            "author_score5": sum(x["author_score"] == 5 for x in sub) / len(sub),
            "gpt4o_exact": rate("gpt4o_exact", sub),
        }
    print(json.dumps(summary, indent=1))

    out = Path(args.out or HERE / f"results_{args.model.replace(':', '-').split('/')[-1]}.json")
    out.write_text(json.dumps({"model": args.model, "summary": summary, "rows": results}, indent=1, ensure_ascii=False),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
