"""
server.py — Flask API server for MedAgent
Run with: python3 server.py
"""
from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import time
import threading
import base64
import agent
import rag
import cv_engine

app = Flask(__name__, static_folder=".")
CORS(app)


# ── Deployment configuration ───────────────────────────────────────────────────
# Everything here is off by default, so running locally behaves exactly as
# before: the user supplies their own key and all tools are available.
# A public deployment sets DEMO_MODE=true and provides OPENAI_API_KEY.

def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

DEMO_MODE           = _env_flag("DEMO_MODE")
SERVER_API_KEY      = os.environ.get("OPENAI_API_KEY", "").strip()
RATE_LIMIT_PER_HOUR = int(os.environ.get("RATE_LIMIT_PER_HOUR", "30"))
SEED_ON_BOOT        = int(os.environ.get("SEED_ON_BOOT", "0"))


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


# Per-IP sliding window. Enforced only in demo mode, where the server pays
# for every request. Held in memory, so run a single worker for it to be exact.
_rate_buckets: dict[str, list[float]] = {}

def _rate_limited() -> bool:
    if not DEMO_MODE:
        return False
    now    = time.time()
    bucket = _rate_buckets.setdefault(_client_ip(), [])
    cutoff = now - 3600
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= RATE_LIMIT_PER_HOUR:
        return True
    bucket.append(now)
    return False


def _resolve_api_key(client_key: str) -> tuple[str, str | None]:
    """In demo mode the server's own key is used and the client's is ignored."""
    if DEMO_MODE:
        if not SERVER_API_KEY:
            return "", "Demo mode is enabled but the server has no OPENAI_API_KEY configured."
        return SERVER_API_KEY, None
    if not client_key:
        return "", "OpenAI API key is required."
    return client_key, None


SEED_STATUS = {"running": False, "done": False, "error": None}

def _maybe_seed_on_boot() -> None:
    """
    Hosts with an ephemeral disk lose ChromaDB on redeploy; refill it.

    Runs on a background thread: seeding downloads the embedding model and
    embeds every record, which takes minutes. Doing that inline would delay
    binding the port, and the platform kills a service that does not listen
    quickly ("No open ports detected").
    """
    if SEED_ON_BOOT <= 0:
        return

    def _run() -> None:
        SEED_STATUS["running"] = True
        try:
            import seed_database
            seed_database.seed_database(SEED_ON_BOOT)
            SEED_STATUS["done"] = True
        except Exception as e:
            SEED_STATUS["error"] = str(e)
            print(f"[WARN] Seed on boot failed: {e}")
        finally:
            SEED_STATUS["running"] = False

    threading.Thread(target=_run, name="seed-on-boot", daemon=True).start()

_maybe_seed_on_boot()

# Per-session conversation history, persisted to disk so the agent
# remembers conversations across server restarts.
_SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_history.json")

def _load_sessions() -> dict:
    if os.path.exists(_SESSION_PATH):
        try:
            with open(_SESSION_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_sessions() -> None:
    tmp_path = _SESSION_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(SESSION_HISTORY, f, indent=2)
        os.replace(tmp_path, _SESSION_PATH)
    except Exception as e:
        print(f"[WARN] Could not save session_history.json: {e}")

SESSION_HISTORY: dict[str, list[dict]] = _load_sessions()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    client_key = data.get("api_key", "").strip()
    patient_id = data.get("patient_id", "default").strip()
    message    = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if _rate_limited():
        return jsonify({
            "error": f"Demo rate limit reached ({RATE_LIMIT_PER_HOUR} messages/hour). "
                     "Please try again later, or run MedAgent locally with your own API key."
        }), 429

    api_key, key_error = _resolve_api_key(client_key)
    if key_error:
        return jsonify({"error": key_error}), 400

    # Retrieve or create session history
    history = SESSION_HISTORY.setdefault(patient_id, [])

    try:
        result = agent.run_agent(api_key, patient_id, message, history)
    except Exception as e:
        err_msg = str(e)
        # Surface friendly OpenAI errors
        if "api_key" in err_msg.lower() or "authentication" in err_msg.lower():
            return jsonify({"error": "Invalid OpenAI API key. Please check your key."}), 401
        return jsonify({"error": err_msg}), 500

    # Append this turn to history
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": result["reply"]})

    # Keep history bounded (last 20 turns = 40 messages)
    if len(history) > 40:
        SESSION_HISTORY[patient_id] = history[-40:]

    _save_sessions()

    return jsonify(result)


@app.route("/healthz", methods=["GET"])
def healthz():
    """Liveness probe. Returns OK as soon as the app is listening, even while
    the background seed is still running."""
    return jsonify({"status": "ok", "seeding": SEED_STATUS}), 200


@app.route("/api/config", methods=["GET"])
def client_config():
    """Lets the UI adapt: hide the key field and delete tool in a public demo."""
    return jsonify({
        "demo_mode":           DEMO_MODE,
        "requires_api_key":    not DEMO_MODE,
        "destructive_enabled": not DEMO_MODE,
        "rate_limit_per_hour": RATE_LIMIT_PER_HOUR if DEMO_MODE else None,
    })


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    return jsonify(rag.get_stats())


@app.route("/api/analytics", methods=["GET"])
def analytics():
    return jsonify(rag.get_aggregate_stats())


@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient(patient_id: str):
    record = agent.PATIENT_STORE.get(patient_id)
    if not record:
        return jsonify({"error": f"Patient '{patient_id}' not found."}), 404
    return jsonify(record)


@app.route("/api/lab/<patient_id>", methods=["GET"])
def get_lab(patient_id: str):
    labs = agent.LAB_STORE.get(patient_id)
    if not labs:
        return jsonify({"error": f"No lab results for patient '{patient_id}'."}), 404
    return jsonify(labs)


@app.route("/api/patients", methods=["GET"])
def list_patients():
    return jsonify({"patients": list(agent.PATIENT_STORE.keys()), "count": len(agent.PATIENT_STORE)})


@app.route("/api/patients/full", methods=["GET"])
def list_patients_full():
    """All patient records with full fields, for the UI patient browser."""
    patients = rag.get_all_patients()
    return jsonify({"patients": patients, "count": len(patients)})


@app.route("/api/lab/<patient_id>/analyzed", methods=["GET"])
def get_lab_analyzed(patient_id: str):
    """Latest lab entry classified against normal ranges (same shape as /api/chat's lab_results)."""
    labs = agent.LAB_STORE.get(patient_id)
    if not labs:
        return jsonify({"error": f"No lab results for patient '{patient_id}'."}), 404
    latest = labs[-1]
    analyzed = {}
    for k, v in latest.items():
        if k == "timestamp":
            continue
        try:
            analyzed[k] = agent._classify_value(k, float(v))
        except (ValueError, TypeError):
            pass
    return jsonify(analyzed)


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report():
    """
    Extract text from a medical report (image or PDF) so the chat agent
    can analyse it.

    Image branch:  OpenCV preprocess (CLAHE + adaptive threshold + deskew)
                   → GPT-4o Vision extracts all readable content.
    PDF branch:    pypdf extracts text directly (no OpenCV needed).

    Request:  { "type": "image"|"pdf", "file": "<base64>", "api_key": "<sk-...>" }
    Response: { "extracted_text": "...", "method": "image_vision"|"pdf_text",
                "metrics": {...}  (only for images) }
    """
    data       = request.get_json(force=True)
    file_type  = (data.get("type") or "image").lower()
    file_b64   = data.get("file", "")
    client_key = data.get("api_key", "").strip()

    if not file_b64:
        return jsonify({"error": "No file data provided."}), 400

    if _rate_limited():
        return jsonify({
            "error": f"Demo rate limit reached ({RATE_LIMIT_PER_HOUR} requests/hour). "
                     "Please try again later."
        }), 429

    # Strip optional data-URL prefix
    if "," in file_b64:
        file_b64 = file_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(file_b64)
    except Exception:
        return jsonify({"error": "Could not decode file data."}), 400

    # ── PDF branch ────────────────────────────────────────────────────────────
    if file_type == "pdf":
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages_text = []
            for page in reader.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    pages_text.append("")
            extracted = "\n\n".join(pages_text).strip()
            if not extracted:
                return jsonify({
                    "error": "PDF appears to be a scanned image with no embedded text. "
                             "Please upload it as an image instead so OpenCV + Vision can read it."
                }), 400
            return jsonify({
                "extracted_text": extracted,
                "method": "pdf_text",
                "pages": len(reader.pages),
            })
        except Exception as e:
            return jsonify({"error": f"PDF parsing failed: {e}"}), 500

    # ── Image branch (OpenCV + GPT-4o Vision) ────────────────────────────────
    api_key, key_error = _resolve_api_key(client_key)
    if key_error:
        return jsonify({"error": "OpenAI API key is required for image reports."}), 400

    try:
        cleaned_data_url, metrics = cv_engine.preprocess_report_image(raw_bytes)

        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=1)

        prompt = (
            "You are extracting content from a medical report image that has already "
            "been preprocessed (grayscale, contrast-enhanced, deskewed). "
            "Transcribe ALL readable text exactly as it appears, preserving line breaks. "
            "Pay extra attention to numeric lab values and their units. "
            "Return only the extracted text — no commentary, no markdown."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": cleaned_data_url, "detail": "high"}},
                ],
            }],
            max_tokens=1024,
            temperature=0,
        )

        extracted = (response.choices[0].message.content or "").strip()
        if not extracted:
            return jsonify({"error": "Vision model returned no text."}), 500

        return jsonify({
            "extracted_text": extracted,
            "method": "image_vision",
            "metrics": metrics,
        })
    except Exception as e:
        return jsonify({"error": f"Image report analysis failed: {e}"}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hosting platforms assign a port via $PORT; default to 8080 locally.
    port = int(os.environ.get("PORT", "8080"))
    print("=" * 55)
    print(f"  MedAgent Server — http://localhost:{port}")
    print(f"  Mode: {'DEMO (server-side key, deletes disabled)' if DEMO_MODE else 'LOCAL (bring your own key)'}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
