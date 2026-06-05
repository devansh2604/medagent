"""
agent.py — MedAgent: GPT-4o-mini agentic system with tool calling
The LLM orchestrates ALL tool calls via tool_choice="auto".
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import rag

# ── Persistence paths ─────────────────────────────────────────────────────────
_LAB_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_store.json")

def _load_lab_store() -> dict:
    """Load lab results from disk. Returns empty dict if file doesn't exist."""
    if os.path.exists(_LAB_STORE_PATH):
        try:
            with open(_LAB_STORE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_lab_store() -> None:
    """Persist the current LAB_STORE to disk atomically (write to .tmp then rename)."""
    tmp_path = _LAB_STORE_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(LAB_STORE, f, indent=2)
        os.replace(tmp_path, _LAB_STORE_PATH)   # atomic on POSIX; safe on Windows too
    except Exception as e:
        # Don't crash the request if disk write fails — log and continue
        print(f"[WARN] Could not save lab_store.json: {e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _load_patient_store() -> dict:
    """Seed PATIENT_STORE from ChromaDB so updates work after a restart."""
    try:
        patients = rag.get_all_patients()
        return {p["patient_id"]: p for p in patients}
    except Exception:
        return {}

# ── Stores (loaded from disk/ChromaDB on startup) ─────────────────────────────
PATIENT_STORE: dict[str, dict] = _load_patient_store()
LAB_STORE: dict[str, list[dict]] = _load_lab_store()

# ── Normal ranges for lab analysis ───────────────────────────────────────────
LAB_NORMAL_RANGES = {
    # CBC
    "wbc":         {"min": 4.5,   "max": 11.0,  "unit": "K/µL",    "name": "White Blood Cells"},
    "rbc":         {"min": 4.5,   "max": 5.9,   "unit": "M/µL",    "name": "Red Blood Cells"},
    "hemoglobin":  {"min": 13.5,  "max": 17.5,  "unit": "g/dL",    "name": "Hemoglobin"},
    "platelets":   {"min": 150,   "max": 400,   "unit": "K/µL",    "name": "Platelets"},
    # Metabolic
    "glucose":     {"min": 70,    "max": 99,    "unit": "mg/dL",   "name": "Glucose"},
    "creatinine":  {"min": 0.7,   "max": 1.3,   "unit": "mg/dL",   "name": "Creatinine"},
    "bun":         {"min": 7,     "max": 20,    "unit": "mg/dL",   "name": "BUN"},
    "sodium":      {"min": 136,   "max": 145,   "unit": "mEq/L",   "name": "Sodium"},
    "potassium":   {"min": 3.5,   "max": 5.0,   "unit": "mEq/L",   "name": "Potassium"},
    # Liver
    "alt":         {"min": 7,     "max": 56,    "unit": "U/L",     "name": "ALT"},
    "ast":         {"min": 10,    "max": 40,    "unit": "U/L",     "name": "AST"},
    "bilirubin":   {"min": 0.1,   "max": 1.2,   "unit": "mg/dL",   "name": "Bilirubin"},
    # Lipid
    "total_cholesterol": {"min": 0, "max": 200, "unit": "mg/dL",   "name": "Total Cholesterol"},
    "ldl":         {"min": 0,     "max": 100,   "unit": "mg/dL",   "name": "LDL Cholesterol"},
    "hdl":         {"min": 40,    "max": 999,   "unit": "mg/dL",   "name": "HDL Cholesterol"},
    "triglycerides":{"min": 0,    "max": 150,   "unit": "mg/dL",   "name": "Triglycerides"},
}

def _classify_value(key: str, value: float) -> dict:
    """Classify a lab value as normal/high/low/critical."""
    if key not in LAB_NORMAL_RANGES:
        return {"value": value, "status": "unknown", "normal_range": "N/A", "unit": ""}

    r = LAB_NORMAL_RANGES[key]
    unit = r["unit"]
    normal_range = f"{r['min']}–{r['max']} {unit}"

    # Critical thresholds (rough clinical flags)
    # When min is 0, multiplying by 0.7 gives 0 — no negative lab value is possible,
    # so skip the critical_low check entirely for those markers.
    critical_low  = r["min"] * 0.7 if r["min"] > 0 else None
    critical_high = r["max"] * 1.5

    if critical_low is not None and value < critical_low:
        status = "critical_low"
    elif value > critical_high:
        status = "critical_high"
    elif value < r["min"]:
        status = "low"
    elif value > r["max"]:
        status = "high"
    else:
        status = "normal"

    return {
        "name": r["name"],
        "value": value,
        "unit": unit,
        "normal_range": normal_range,
        "status": status,
    }


# ── Tool definitions for OpenAI ───────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "patient_record_tool",
            "description": (
                "Manages patient records in memory and ChromaDB. "
                "Use action='get' to fetch a patient's full record by patient_id. "
                "Use action='add' to create a new patient record. "
                "Use action='update' to modify a specific field of an existing record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "add", "update"],
                        "description": "'get' fetches full details by patient_id; 'add' creates a new record; 'update' modifies an existing one.",
                    },
                    "patient_id": {
                        "type": "string",
                        "description": "Unique patient ID. For 'add', leave blank to auto-generate. For 'update', required.",
                    },
                    "name":           {"type": "string"},
                    "age":            {"type": "integer"},
                    "gender":         {"type": "string"},
                    "symptoms":       {"type": "string", "description": "Comma-separated list of symptoms."},
                    "medical_history":{"type": "string"},
                    "medications":    {"type": "string"},
                    "allergies":      {"type": "string"},
                    "diagnosis":      {"type": "string", "description": "Suspected or confirmed diagnosis."},
                    "update_field":   {"type": "string", "description": "Field name to update (for action='update')."},
                    "update_value":   {"type": "string", "description": "New value for the field (for action='update')."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search_patients",
            "description": (
                "Semantic vector search across all patient records in ChromaDB. "
                "Call IMMEDIATELY when user says: find, search, show me, similar, "
                "who has, look up, cases, patients with, have you seen. "
                "Returns up to 20 semantically similar patients by default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. 'patients with diabetes and hypertension'.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 20).",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_patients",
            "description": (
                "Returns EVERY patient record stored in ChromaDB — no query needed. "
                "Call this when the user asks to list, show all, or count all patients."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_patients_tool",
            "description": (
                "Handles ALL patient deletion — one, many, or all. "
                "action='single': delete one patient by patient_id. "
                "action='bulk': delete a list of patients by patient_ids array — "
                "  WORKFLOW: first call list_all_patients to get real IDs, then pass those SEED-xxx IDs here. "
                "  Never pass position numbers — always real IDs from list_all_patients. "
                "action='all': wipe every record — ONLY after user explicitly confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["single", "bulk", "all"],
                        "description": "'single' deletes one patient, 'bulk' deletes many, 'all' wipes everything.",
                    },
                    "patient_id": {
                        "type": "string",
                        "description": "Required for action='single'. The exact patient_id to delete.",
                    },
                    "patient_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required for action='bulk'. List of exact SEED-xxx patient_id strings.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lab_test_analysis_tool",
            "description": (
                "Stores and analyzes lab test results. "
                "Use action='store' when the user provides lab values. "
                "Use action='analyze' when the user asks about their lab results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "analyze", "history"],
                        "description": "'store' saves new values, 'analyze' reads the latest entry, 'history' returns all past entries.",
                    },
                    "patient_id": {"type": "string", "description": "Patient ID."},
                    # CBC
                    "wbc":         {"type": "number"},
                    "rbc":         {"type": "number"},
                    "hemoglobin":  {"type": "number"},
                    "platelets":   {"type": "number"},
                    # Metabolic
                    "glucose":     {"type": "number"},
                    "creatinine":  {"type": "number"},
                    "bun":         {"type": "number"},
                    "sodium":      {"type": "number"},
                    "potassium":   {"type": "number"},
                    # Liver
                    "alt":         {"type": "number"},
                    "ast":         {"type": "number"},
                    "bilirubin":   {"type": "number"},
                    # Lipid
                    "total_cholesterol": {"type": "number"},
                    "ldl":         {"type": "number"},
                    "hdl":         {"type": "number"},
                    "triglycerides":{"type": "number"},
                },
                "required": ["action", "patient_id"],
            },
        },
    },
]

# ── Tool executor ─────────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> str:
    """Run the named tool and return a JSON string result."""

    # ── Tool 1: patient_record_tool ──────────────────────────────────────────
    if name == "patient_record_tool":
        action = args.get("action")

        if action == "get":
            pid = args.get("patient_id")
            if not pid:
                return json.dumps({"status": "error", "message": "patient_id is required for action='get'."})
            # Check in-memory store first, then fall back to ChromaDB
            record = PATIENT_STORE.get(pid) or rag.get_patient(pid)
            if not record:
                return json.dumps({"status": "error", "message": f"Patient '{pid}' not found."})
            # Ensure it's in PATIENT_STORE for the dashboard to pick up
            PATIENT_STORE[pid] = record
            return json.dumps({"status": "success", "action": "get", "patient_id": pid, "record": record})

        if action == "add":
            pid = args.get("patient_id") or f"P{uuid.uuid4().hex[:6].upper()}"
            record = {
                "patient_id": pid,
                "name":            args.get("name", "Unknown"),
                "age":             args.get("age", "N/A"),
                "gender":          args.get("gender", "N/A"),
                "symptoms":        args.get("symptoms", "None"),
                "medical_history": args.get("medical_history", "None"),
                "medications":     args.get("medications", "None"),
                "allergies":       args.get("allergies", "None"),
                "diagnosis":       args.get("diagnosis", "Pending"),
            }
            PATIENT_STORE[pid] = record
            rag.upsert_patient(pid, record)
            return json.dumps({"status": "success", "action": "add", "patient_id": pid, "record": record})

        elif action == "update":
            pid = args.get("patient_id")
            if not pid or pid not in PATIENT_STORE:
                return json.dumps({"status": "error", "message": f"Patient {pid} not found."})
            field = args.get("update_field")
            value = args.get("update_value")
            if not field:
                return json.dumps({"status": "error", "message": "update_field is required for action='update'."})
            PATIENT_STORE[pid][field] = value
            rag.upsert_patient(pid, PATIENT_STORE[pid])
            return json.dumps({"status": "success", "action": "update", "patient_id": pid, "field": field, "new_value": value})

        return json.dumps({"status": "error", "message": "Unknown action."})

    # ── Tool 2: rag_search_patients ──────────────────────────────────────────
    elif name == "rag_search_patients":
        query = args.get("query", "")
        n = args.get("n_results", 20)
        results = rag.search_patients(query, n_results=n)
        return json.dumps({"status": "success", "results": results, "count": len(results)})

    # ── Tool 3: list_all_patients ─────────────────────────────────────────────
    elif name == "list_all_patients":
        patients = rag.get_all_patients()
        slim = [{"patient_id": p["patient_id"], "name": p["name"],
                 "age": p.get("age", "N/A"), "gender": p.get("gender", "N/A"),
                 "diagnosis": p.get("diagnosis", "N/A")} for p in patients]
        # Pre-format as numbered text so the LLM can pass it through verbatim
        lines = [f"{i+1}. {p['name']} (ID: {p['patient_id']}, Age: {p['age']}, Gender: {p['gender']}, Dx: {p['diagnosis']})"
                 for i, p in enumerate(slim)]
        formatted = "\n".join(lines)
        return json.dumps({"status": "success", "count": len(slim), "patients": slim, "formatted_list": formatted})

    # ── Tool 4: delete_patients_tool (single / bulk / all) ───────────────────
    elif name == "delete_patients_tool":
        action = args.get("action")

        if action == "single":
            pid = args.get("patient_id")
            if not pid:
                return json.dumps({"status": "error", "message": "patient_id is required for action='single'."})
            d_rag   = rag.delete_patient(pid)
            d_store = PATIENT_STORE.pop(pid, None) is not None
            LAB_STORE.pop(pid, None)
            _save_lab_store()
            if d_rag or d_store:
                return json.dumps({"status": "success", "message": f"Patient {pid} deleted."})
            return json.dumps({"status": "error", "message": f"Patient {pid} not found."})

        elif action == "bulk":
            pids = args.get("patient_ids", [])
            if not pids:
                return json.dumps({"status": "error", "message": "patient_ids list is required for action='bulk'."})
            deleted, not_found = [], []
            for pid in pids:
                if rag.delete_patient(pid) or PATIENT_STORE.pop(pid, None) is not None:
                    deleted.append(pid)
                else:
                    not_found.append(pid)
                LAB_STORE.pop(pid, None)   # clean up any lab data too
            _save_lab_store()
            return json.dumps({
                "status": "success",
                "deleted_count": len(deleted),
                "not_found_count": len(not_found),
                "message": f"Deleted {len(deleted)} patients. {len(not_found)} IDs not found.",
            })

        elif action == "all":
            deleted_count = rag.delete_all_patients()
            PATIENT_STORE.clear()
            LAB_STORE.clear()
            _save_lab_store()   # persist the cleared state
            return json.dumps({"status": "success", "message": "All patient records deleted.", "deleted_count": deleted_count})

        return json.dumps({"status": "error", "message": "Unknown action. Use 'single', 'bulk', or 'all'."})

    # ── Tool 5: lab_test_analysis_tool ───────────────────────────────────────
    elif name == "lab_test_analysis_tool":
        action = args.get("action")
        pid = args.get("patient_id", "unknown")

        lab_keys = [
            "wbc","rbc","hemoglobin","platelets",
            "glucose","creatinine","bun","sodium","potassium",
            "alt","ast","bilirubin",
            "total_cholesterol","ldl","hdl","triglycerides",
        ]

        if action == "store":
            labs = {k: args[k] for k in lab_keys if k in args}
            if not labs:
                return json.dumps({"status": "error", "message": "No lab values provided."})
            labs["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if pid not in LAB_STORE:
                LAB_STORE[pid] = []
            LAB_STORE[pid].append(labs)
            _save_lab_store()   # persist to disk immediately
            return json.dumps({"status": "success", "action": "store", "patient_id": pid,
                                "stored_fields": [k for k in labs if k != "timestamp"],
                                "timestamp": labs["timestamp"],
                                "total_entries": len(LAB_STORE[pid])})

        elif action == "analyze":
            if pid not in LAB_STORE or not LAB_STORE[pid]:
                return json.dumps({"status": "error", "message": f"No lab results found for patient {pid}."})
            latest = LAB_STORE[pid][-1]
            analysis = {}
            for k, v in latest.items():
                if k == "timestamp":
                    continue
                try:
                    analysis[k] = _classify_value(k, float(v))
                except (ValueError, TypeError):
                    pass
            return json.dumps({"status": "success", "action": "analyze", "patient_id": pid,
                                "timestamp": latest.get("timestamp", "Unknown"),
                                "analysis": analysis})

        elif action == "history":
            if pid not in LAB_STORE or not LAB_STORE[pid]:
                return json.dumps({"status": "error", "message": f"No lab history for patient {pid}."})
            entries = LAB_STORE[pid]
            # Build per-metric trend summary so LLM can answer "how many times did X change"
            all_keys = set()
            for e in entries:
                all_keys.update(k for k in e if k != "timestamp")
            trends = {}
            for key in sorted(all_keys):
                vals = [{"timestamp": e["timestamp"], "value": e[key]}
                        for e in entries if key in e]
                if not vals:
                    continue
                try:
                    numeric = [float(v["value"]) for v in vals]
                    changes = sum(1 for i in range(1, len(numeric)) if numeric[i] != numeric[i-1])
                    trend_dir = ("increasing" if numeric[-1] > numeric[0]
                                 else "decreasing" if numeric[-1] < numeric[0]
                                 else "stable")
                    r = LAB_NORMAL_RANGES.get(key, {})
                    trends[key] = {
                        "readings": len(vals),
                        "times_changed": changes,
                        "trend": trend_dir,
                        "first":  numeric[0],
                        "latest": numeric[-1],
                        "delta":  round(numeric[-1] - numeric[0], 4),
                        "min_recorded": min(numeric),
                        "max_recorded": max(numeric),
                        "normal_range": f"{r.get('min','?')}–{r.get('max','?')} {r.get('unit','')}" if r else "N/A",
                        "values": vals,
                    }
                except (ValueError, TypeError):
                    trends[key] = {"readings": len(vals), "values": vals}
            return json.dumps({"status": "success", "action": "history", "patient_id": pid,
                                "total_entries": len(entries), "history": entries,
                                "trends": trends})

        return json.dumps({"status": "error", "message": "Unknown action."})

    return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})


# ── Main agent entry point ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedAgent — a compassionate, knowledgeable medical AI assistant.

YOUR TOOLS — you have exactly these 5 tools, no more, no less:
  1. patient_record_tool   — get full record, add, or update a patient
  2. rag_search_patients   — semantic search by symptoms / condition / similarity
  3. list_all_patients     — returns ALL patients with name, id, age, gender, diagnosis
  4. delete_patients_tool  — delete one patient, a range of patients, or all patients
  5. lab_test_analysis_tool — store / analyze / history for lab results

WHEN TO CALL EACH TOOL:
- patient_record_tool (action=get)    → user asks for details/full record/history of a specific patient.
  WORKFLOW for "get details of patient N" or "load them":
    Step 1: call list_all_patients to get the numbered list and find the real patient_id at position N
    Step 2: call patient_record_tool(action='get', patient_id=<real ID from step 1>)
  This 2-step flow is REQUIRED — never skip step 2, it loads the full record into the dashboard.
  If the patient_id is already known (user said the ID directly), skip step 1.
- patient_record_tool (action=add)    → user gives name, age, gender, symptoms to register a new patient
- patient_record_tool (action=update) → user gives an updated field for an existing patient
- rag_search_patients                 → user says find, search, similar, who has, patients with
- list_all_patients                   → user says "list all patients", "show all", OR asks for stats, counts, average age, demographics
- delete_patients_tool                → any deletion request; choose the right action:
    action='single'  → "delete patient X" — pass patient_id
    action='bulk'    → "delete patients 50-100" or a group — WORKFLOW:
                         Step 1: call list_all_patients to get the numbered list with real SEED-xxx IDs
                         Step 2: extract the IDs at the requested list positions
                         Step 3: call delete_patients_tool(action='bulk', patient_ids=[...real IDs...])
                         NEVER pass position numbers like "100" — always pass real SEED-xxx IDs
    action='all'     → "delete everything" / "wipe all" — ONLY after user explicitly confirms
- lab_test_analysis_tool (action=store)   → user provides lab values
- lab_test_analysis_tool (action=analyze) → user asks about their current/latest lab results
- lab_test_analysis_tool (action=history) → user asks about trends, changes, "how many times", "has X changed", "compare", "over time", "previous results"
  The history response includes a "trends" object with per-metric: readings count, times_changed, trend direction, first/latest values, delta. Use this to answer trend questions precisely.

NO TOOL NEEDED — answer directly:
- "what tools do you have" / "list your tools" / "what can you do" → list all 5 tools above by name, no tool call
- Diagnosis reasoning → use medical knowledge, always say "possible" or "suspected"
- Mental health support → read emotional tone, respond with empathy
- CRISIS SIGNALS (suicide, self-harm, hopelessness) → immediately say "Please call or text 988 (Suicide & Crisis Lifeline) — you are not alone."
- General medical questions, medication interactions, side effects → answer from training knowledge

FORMATTING RULES:
- Never use LaTeX or math notation (no \\frac, \\text, \\[, etc.) — write all calculations in plain text
- Use plain language; avoid unnecessary jargon
- When list_all_patients returns results, the tool includes a "formatted_list" field. Copy it verbatim into your reply — do NOT reformat, summarize, or truncate it. Start your reply with "Here are all X patients:" then paste the formatted_list exactly.

ALWAYS:
- Recommend consulting a licensed physician for all medical decisions
- Be warm, empathetic, and non-judgmental
- Acknowledge emotions before giving medical information
"""


def run_agent(api_key: str, patient_id: str, message: str, history: list[dict]) -> dict:
    """
    Run one turn of the agentic loop.

    Parameters
    ----------
    api_key    : OpenAI API key from the request
    patient_id : current patient session ID
    message    : user's latest message
    history    : list of prior {role, content} dicts for this session

    Returns
    -------
    dict with keys: reply, patient_record, lab_results, tools_called
    """
    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": message})

    tools_called = []
    touched_pids = []   # patient_ids actually written/read this turn
    reply_text = ""     # ensure reply_text is always defined

    # ── Agentic loop ─────────────────────────────────────────────────────────
    MAX_ITERATIONS = 15
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.4,
            max_tokens=4096,
        )

        msg = response.choices[0].message

        # No tool call → final answer
        if not msg.tool_calls:
            reply_text = msg.content or ""
            break

        # Append assistant's tool-call message
        messages.append(msg)

        # Execute each tool call
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tools_called.append(tool_name)

            # Inject session patient_id only for tools that actually use it
            TOOLS_USING_PATIENT_ID = {"patient_record_tool", "lab_test_analysis_tool"}
            if tool_name in TOOLS_USING_PATIENT_ID and not tool_args.get("patient_id"):
                tool_args["patient_id"] = patient_id

            result = _execute_tool(tool_name, tool_args)

            # Track which patient IDs were touched this turn
            try:
                result_data = json.loads(result)
                pid_used = result_data.get("patient_id") or tool_args.get("patient_id")
                if pid_used:
                    touched_pids.append(pid_used)
            except Exception:
                pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # ── Build response payload ────────────────────────────────────────────────
    if touched_pids:
        # A specific patient was looked up/modified this turn — show ONLY their data.
        # Never fall back to the session patient; that would mix up different patients' labs.
        lookup_pid     = touched_pids[-1]
        patient_record = PATIENT_STORE.get(lookup_pid)
        lab_results    = LAB_STORE.get(lookup_pid)       # None if this patient has no labs yet
    else:
        # No specific patient touched → fall back to session patient
        lookup_pid     = patient_id
        patient_record = PATIENT_STORE.get(patient_id)
        lab_results    = LAB_STORE.get(patient_id)

    # Attach classified analysis of the most recent lab entry if labs exist
    if lab_results:
        latest = lab_results[-1]   # lab_results is now a list of timestamped entries
        analyzed = {}
        for k, v in latest.items():
            if k == "timestamp":
                continue
            try:
                analyzed[k] = _classify_value(k, float(v))
            except (ValueError, TypeError):
                pass   # skip any corrupted / non-numeric lab value
    else:
        analyzed = None

    return {
        "reply":          reply_text,
        "patient_record": patient_record,
        "lab_results":    analyzed,
        "tools_called":   list(set(tools_called)),
    }
