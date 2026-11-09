"""Local, model-agnostic adapter for SOP-Bench aircraft_inspection.

No API calls or downloads in this module. Official tools are mock CSV-backed APIs.
The finish_inspection function is our transport adapter, not an original tool.
"""
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import time

REVISION = "92633eae202fbf66cceb374c6ad55da41ca693b6"
PROTOCOL_VERSION = "json_transport_v2"
EXPECTED_HASHES = {
    "metadata.json": "8ede3a986812c85fea89ba6f56d09fc174142af346d361c80327f3b1933276be",
    "sop.txt": "83e22b9604827ba07b2a1790580a6af1a5a5427d6a80204cdc3303fd47bb941b",
    "tools.py": "571b20ffe2b6ea25e1c3e86798b2f7640efbbe680e05995473f031f5e32171ec",
    "toolspecs.json": "12ae40830defdbe563490852d5d55466d42fbbc895ceae72a82de1141c9809d3",
    "test_set_with_outputs.csv": "20646ee23b8c09a93945eeac0a47bc851bd32564bbf9584ab805e26d3367d0f2",
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def normalize(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    return None if value in ("", "nan", "none", "null") else value


def prepare(data_dir):
    data_dir = Path(data_dir)
    for name, expected in EXPECTED_HASHES.items():
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == expected, name
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((data_dir / "test_set_with_outputs.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 112 and len({r["aircraft_id"] for r in rows}) == 112
    spec = importlib.util.spec_from_file_location("audited_sop_aircraft_tools", data_dir / "tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manager = module.AircraftInspectionManager()
    toolspecs = json.loads((data_dir / "toolspecs.json").read_text(encoding="utf-8"))
    names = {s["toolSpec"]["name"] for s in toolspecs}
    cases, references = [], {}
    for row in sorted(rows, key=lambda r: r["aircraft_id"]):
        inputs = {k: row[k] if row[k] != "" else None for k in metadata["input_columns"]}
        for key in ("component_weight", "expected_component_weight"):
            if inputs[key] is not None:
                inputs[key] = float(inputs[key])
        cases.append({"case_id": row["aircraft_id"], "inputs": inputs})
        references[row["aircraft_id"]] = {k: row[k] for k in metadata["output_columns"]}
    assert not (set(metadata["input_columns"]) & set(metadata["output_columns"]))
    return cases, references, manager, names, metadata, toolspecs


def execute_command(command, case, manager, names):
    """Only parse literals and dispatch allowlisted mock APIs; never eval/exec model text."""
    if not isinstance(command, str):
        raise ValueError("command must be a string")
    node = ast.parse(command.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("Expected one named function call")
    if node.args or any(k.arg is None for k in node.keywords):
        raise ValueError("Use explicit keyword arguments, no positional arguments or **kwargs")
    if len({k.arg for k in node.keywords}) != len(node.keywords):
        raise ValueError("Duplicate keyword arguments")
    arguments = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    name = node.func.id
    if name == "finish_inspection":
        if set(arguments) != {"result"} or not isinstance(arguments["result"], dict):
            raise ValueError("finish_inspection(result={...}) expected")
        return name, arguments, {"finished": True, "result": arguments["result"]}
    if name not in names:
        raise ValueError("Unknown tool: " + name)
    if arguments.get("aircraft_id") != case["case_id"]:
        raise ValueError("Tool must operate on the current aircraft_id")
    inspect.signature(getattr(manager, name)).bind(**arguments)
    observation = manager.process_tool_call(name, arguments)
    return name, arguments, json_safe(observation)


def system_prompt(data_dir, metadata, toolspecs):
    sop = (Path(data_dir) / "sop.txt").read_text(encoding="utf-8")
    return (
        "You are an inspection workflow agent in an offline mock environment. Follow the supplied SOP.\n"
        "Transport protocol for this reconstructed runner: on each turn output ONLY a JSON object "
        "with string keys reason and command. command must contain exactly one allowed Python-style "
        "function call using explicit keyword arguments and literal values. Never invent tool results. "
        "After a tool call, wait for its observation. Do not call tools for other aircraft. "
        "If a tool reports an error, use it as feedback; do not claim it succeeded.\n"
        "For final submission only, replace the SOP's XML reporting wrapper by "
        "finish_inspection(result={...}). Include aircraft_id and these result fields: "
        + json.dumps(metadata["output_columns"]) + ". Use Python literal None for inapplicable fields "
        "inside the command string. This finish function is a runner adapter, not a real operation.\n"
        "Other SOP decision logic is unchanged.\n\nSOP:\n" + sop
        + "\n\nAvailable mock tool specifications:\n" + json.dumps(toolspecs, ensure_ascii=False)
        + transport_reminder(metadata["output_columns"])
    )


def transport_reminder(output_columns):
    example_result = {"aircraft_id": "<current aircraft_id>",
                      **{k: "<value from the corresponding tool observation>" for k in output_columns}}
    example = {"reason": "Summarize the observations supporting this report.",
               "command": "finish_inspection(result=" + repr(example_result) + ")"}
    return (
        "\n\nFINAL TRANSPORT REMINDER (also applies to the last turn):\n"
        "Every response MUST be a JSON object with string reason and command. "
        "A bare function call, XML or Markdown is not a valid response. "
        "finish_inspection(...) must be inside the JSON command STRING, never outside JSON. "
        "Example of the final transport shape only; replace all placeholders using actual observations:\n"
        + json.dumps(example, ensure_ascii=False)
        + "\nInclude all listed result keys, including cross_check_response. "
        "Do not invent success or fill unobserved results with success. "
        "Only use None for genuinely inapplicable fields, not as a substitute for a failed/missing check. "
        "This reminder changes the transport instructions, not the SOP or scoring rules.\n"
    )


def run_case(case, manager, names, sys_prompt, generate, run_dir, max_turns=12):
    run_dir = Path(run_dir)
    result_path = run_dir / "responses" / (case["case_id"] + ".json")
    user_text = "Inspect this aircraft using the SOP and mock tools.\n" + json.dumps(case["inputs"], ensure_ascii=False)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]
    prompt_blob = json.dumps(messages, ensure_ascii=False)
    initial_hash = hashlib.sha256(prompt_blob.encode()).hexdigest()
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        assert existing["inputs"] == case["inputs"]
        if existing.get("initial_messages_sha256") != initial_hash:
            raise ValueError("Prompt changed: create a new run directory; do not reuse older results.")
        return existing
    record = {"case_id": case["case_id"], "inputs": case["inputs"], "trace": [],
              "finished": False, "final_result": None, "error": None}
    record["initial_messages_sha256"] = initial_hash
    (run_dir / "prompts" / (case["case_id"] + ".json")).write_text(prompt_blob, encoding="utf-8")
    try:
        for turn in range(1, max_turns + 1):
            generated = generate(messages)
            entry = {"turn": turn, **generated, "format_valid": False, "tool_error": None}
            record["trace"].append(entry)
            raw = generated["raw_output"]
            messages.append({"role": "assistant", "content": raw})
            try:
                obj = json.loads(raw.strip())
                if not isinstance(obj, dict) or not all(isinstance(obj.get(k), str) for k in ("reason", "command")):
                    raise ValueError("Expected JSON with string reason and command")
                entry.update(format_valid=True, reason=obj["reason"], command=obj["command"])
                name, arguments, observation = execute_command(obj["command"], case, manager, names)
                entry.update(tool=name, arguments=arguments, observation=observation)
                if name == "finish_inspection":
                    record.update(finished=True, final_result=json_safe(observation["result"]))
                    break
                feedback = {"observation": observation}
            except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                entry["tool_error"] = type(exc).__name__ + ": " + str(exc)
                feedback = {"error": entry["tool_error"]}
            messages.append({"role": "user", "content": "Mock tool feedback: " + json.dumps(feedback, ensure_ascii=False, allow_nan=False)})
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        raise
    if not record["finished"]:
        record["error"] = "max_turns_reached"
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record


def score_result(record, reference, output_columns):
    final = record.get("final_result") or {}
    field_matches = {k: k in final and normalize(final[k]) == normalize(reference[k]) for k in output_columns}
    trace = record["trace"]
    executed = [t.get("tool") for t in trace if t.get("tool") and not t.get("tool_error")]
    required_initial = ["VerifyAircraftClearance", "VerifyMechanicalComponents", "VerifyElectricalSystems"]
    initial_order_ok = executed[:3] == required_initial
    return {"case_id": record["case_id"], "finished": record["finished"],
            "all_fields_match": bool(record["finished"] and all(field_matches.values())),
            "field_matches": field_matches, "initial_three_order_ok": initial_order_ok,
            "valid_format_turns": sum(t["format_valid"] for t in trace), "turns": len(trace),
            "tool_errors": sum(t.get("tool_error") is not None for t in trace),
            "note": "Final-field matching is a supplementary task metric, not LLM4IAS command accuracy or full SOP compliance."}
