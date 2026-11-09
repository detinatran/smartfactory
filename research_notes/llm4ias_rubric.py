"""F/C/R review helpers. No model calls, no execution/repair of generated commands."""
import ast
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random

RUBRIC_VERSION = "2409_v1_user_accepted_2026-10-29"
CONFLICT_CASES = {"inspection_station_017"}
# Public prompt function signatures; used only by evaluator, never by inference.
ARGUMENT_NAMES = {
    "conveyor_1_run": ("direction", "time"), "conveyor_2_run": ("direction", "time"),
    "conveyor_3_run": ("direction", "time"), "conveyor_4_run": ("direction", "time"),
    "switch_actuate": ("action",), "alert_to_supervisor": ("reason",),
    "RFID_read_workpiece_info": (), "transport_robot_request": (),
    "release_holder_H1": (), "release_holder_H2": (), "release_holder_H3": (),
    "emergency_stop": (),
}


def rubric_parse(raw):
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError, AttributeError):
        return {"F": 0, "command": None, "reason": None}
    if not isinstance(obj, dict):
        return {"F": 0, "command": None, "reason": None}
    good = all(isinstance(obj.get(k), str) for k in ("reason", "command"))
    return {"F": int(good), "command": obj.get("command"), "reason": obj.get("reason")}


def strict_ast(command):
    try:
        node = ast.parse(command.strip(), mode="eval").body
    except (SyntaxError, ValueError, TypeError, AttributeError):
        return None
    return ast.dump(node, annotate_fields=True, include_attributes=False) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def normalized_literal(value):
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", Decimal(str(value)))
    if isinstance(value, str) or value is None:
        return (type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(normalized_literal(v) for v in value))
    raise ValueError("Unsupported argument literal")


def command_signature(command):
    try:
        node = ast.parse(command.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        if any(k.arg is None for k in node.keywords) or len({k.arg for k in node.keywords}) != len(node.keywords):
            return None
        args = [normalized_literal(ast.literal_eval(a)) for a in node.args]
        kwargs = {k.arg: normalized_literal(ast.literal_eval(k.value)) for k in node.keywords}
        if node.func.id in ARGUMENT_NAMES:
            params = ARGUMENT_NAMES[node.func.id]
            if len(args) > len(params) or any(k not in params for k in kwargs):
                return None
            bound = dict(zip(params,args))
            if set(bound) & set(kwargs):
                return None
            bound.update(kwargs)
            if set(bound) != set(params):
                return None
            return (node.func.id, tuple(bound[p] for p in params), ())
        return (node.func.id, tuple(args), tuple(sorted(kwargs.items())))
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return None


def suggest_C(case_id, raw, reference):
    parsed = rubric_parse(raw)
    if case_id in CONFLICT_CASES:
        return "U", "Reference conflicts with event/ref_reason; keep U for both sources."
    if not parsed["F"]:
        return "U", "Invalid format: inspect original text for verbatim intent; no repair."
    predicted, target = command_signature(parsed["command"]), command_signature(reference)
    if target is None:
        return "U", "Reference cannot be parsed by this helper."
    if predicted == target:
        return "1", "Single reference call matches after literal normalization; reviewer must confirm."
    if predicted and predicted[0] == target[0] == "alert_to_supervisor":
        return "U", "Review alert message semantics against device/fault; never score by tool name alone."
    return "0", "Different action/operational arguments or unsupported/multiple call; reviewer must confirm."


def export_review(cases, results, references, run_dir):
    """Create 44 masked items and suggestions separately. Preserve manual CSV on rerun."""
    run_dir = Path(run_dir)
    assert len(cases) == len(results) == 22
    new = {r["case_id"]: r for r in results}
    assert len(new) == 22 and set(new) == {c["case_id"] for c in cases}
    rng = random.Random(2409)
    items, key, manual, auto = [], [], [], []
    md = ["# Review 22 test / 44 responses\n\nRead rubric_2409_v1.md first. "
          "A/B source assignments vary per case. Do not open source_key.csv before review. "
          "Source masking cannot guarantee the reviewer cannot guess from style.\n"]
    for case in cases:
        cid = case["case_id"]
        ref = references[cid]
        assert new[cid].get("raw_output") is not None and not new[cid].get("error")
        assert new[cid]["prompt_sha256"] == case["prompt_sha256"]
        pair = [("new_run", new[cid]["raw_output"]), ("author_saved_run", ref["author_raw"])]
        assert all(isinstance(raw, str) for _,raw in pair)
        rng.shuffle(pair)
        md.append(f"\n## {cid} — {case['type']}\n\nReference: `{ref['correct_command']}`\n\n"
                  + ("REFERENCE CONFLICT: keep C=U.\n\n" if cid in CONFLICT_CASES else "")
                  + "Event log:\n\n```text\n" + case["event_log"] + "\n```\n")
        for label, (source, raw) in zip(("A", "B"), pair):
            item_id = cid + "_" + label
            parsed = rubric_parse(raw)
            suggestion, why = suggest_C(cid, raw, ref["correct_command"])
            strict = strict_ast(parsed["command"])
            matched = bool(parsed["F"] and strict is not None and strict == strict_ast(ref["correct_command"]))
            items.append({"item_id": item_id, "case_id": cid, "type": case["type"],
                          "event_log": case["event_log"], "reference_command": ref["correct_command"],
                          "reference_reason": ref["reference_reason"], "response_raw": raw,
                          "F_auto": parsed["F"], "reference_conflict": cid in CONFLICT_CASES})
            key.append({"item_id": item_id, "source": source})
            manual.append({"item_id": item_id, "C_final_0_1_U": "", "R_final_1_to_5_U": "",
                           "evidence_mode": "", "reviewer": "", "review_notes": ""})
            auto.append({"item_id": item_id, "case_id": cid, "type": case["type"], "source": source,
                         "F": parsed["F"], "strict_ast_match": int(matched),
                         "C_suggested": suggestion, "suggestion_note": why,
                         "command": parsed["command"], "reason": parsed["reason"]})
            md.append(f"\n### Response {label}\n\n```text\n{raw}\n```\n")
    (run_dir / "review_blind.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in items), encoding="utf-8")
    (run_dir / "review_blind.md").write_text("".join(md), encoding="utf-8")
    for name, rows in (("source_key.csv",key),("auto_suggestions.csv",auto),("review_scores.csv",manual)):
        p = run_dir / name
        if name == "review_scores.csv" and p.exists():
            continue
        with p.open("w", newline="", encoding="utf-8-sig") as file:
            w = csv.DictWriter(file, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    context = {"rubric_version": RUBRIC_VERSION, "items": len(items), "case_count": len(cases),
               "state": "C suggestions are provisional; no R labels generated; manual scores pending"}
    (run_dir / "review_manifest.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return items, auto


def summarize_review(items, auto, score_path):
    with Path(score_path).open(encoding="utf-8-sig", newline="") as file:
        manual_rows = list(csv.DictReader(file))
    manual = {r["item_id"]:r for r in manual_rows}
    expected_ids = {i["item_id"] for i in items}
    assert len(manual_rows) == len(manual) == len(items) and set(manual) == expected_ids, "Missing/duplicate/unknown review IDs"
    auto_map = {a["item_id"]: a for a in auto}
    source = {a["item_id"]: a["source"] for a in auto}
    checked = []
    for item in items:
        row = manual[item["item_id"]]
        c = row["C_final_0_1_U"].strip().upper()
        r = row["R_final_1_to_5_U"].strip().upper()
        if c not in ("", "0", "1", "U") or r not in ("", "1", "2", "3", "4", "5", "U"):
            raise ValueError("Invalid manual score: " + item["item_id"])
        if item["reference_conflict"] and c not in ("", "U"):
            raise ValueError("Reference conflict is U for both sources under rubric v1: " + item["item_id"])
        if not item["F_auto"] and (c in ("0", "1") or r in ("1", "2", "3", "4", "5")) and row["evidence_mode"] != "verbatim_intent":
            raise ValueError("Invalid-format content needs evidence_mode=verbatim_intent: " + item["item_id"])
        if (c and c != auto_map[item["item_id"]]["C_suggested"]) and not row["review_notes"].strip():
            raise ValueError("Document why manual C differs from suggestion: " + item["item_id"])
        checked.append({**item,"source":source[item["item_id"]],"C_final":c,"R_final":r})
    summary = {"rubric_version":RUBRIC_VERSION,"manual_scores_complete":all(x["C_final"] and x["R_final"] for x in checked),"sources":{}}
    for label in ("new_run", "author_saved_run"):
        summary["sources"][label] = {}
        for group in ("all", "SOP", "Unexpected Events"):
            subset=[x for x in checked if x["source"]==label and (group=="all" or x["type"]==group)]
            n=len(subset);k=sum(x["C_final"]=="1" for x in subset);w=sum(x["C_final"]=="0" for x in subset)
            u=sum(x["C_final"]=="U" for x in subset);pending=sum(not x["C_final"] for x in subset)
            reasons=[int(x["R_final"]) for x in subset if x["R_final"] in ("1","2","3","4","5")]
            e_correct=sum(x["F_auto"]==1 and x["C_final"]=="1" for x in subset)
            e_unknown=sum(x["F_auto"]==1 and x["C_final"] in ("","U") for x in subset)
            summary["sources"][label][group]={"n":n,"F_valid":sum(x["F_auto"] for x in subset),
                "strict_ast_matches":sum(auto_map[x["item_id"]]["strict_ast_match"] for x in subset),
                "C_manual_correct":k,"C_manual_wrong":w,"C_manual_U":u,"C_pending":pending,
                "C_accuracy_bounds":[k/n,(k+u+pending)/n],"C_determined_coverage":(k+w)/n,
                "C_accuracy_on_determined":k/(k+w) if k+w else None,
                "R_mean":sum(reasons)/len(reasons) if reasons else None,"R_scored_count":len(reasons),
                "R_distribution":{str(value):reasons.count(value) for value in range(1,6)},
                "R_U":sum(x["R_final"]=="U" for x in subset),"R_pending":sum(not x["R_final"] for x in subset),
                "E_correct":e_correct,"E_unknown_or_pending":e_unknown,
                "C_suggestions_only":{s:sum(auto_map[x["item_id"]]["C_suggested"]==s for x in subset) for s in ("0","1","U")}}
    return summary
