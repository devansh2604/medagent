"""
server.py — Flask API server for MedAgent
Run with: python3 server.py
"""
from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import agent
import rag
import cv_engine

app = Flask(__name__, static_folder=".")
CORS(app)

# Per-session conversation history (in-memory, keyed by patient_id)
SESSION_HISTORY: dict[str, list[dict]] = {}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    api_key    = data.get("api_key", "").strip()
    patient_id = data.get("patient_id", "default").strip()
    message    = data.get("message", "").strip()

    if not api_key:
        return jsonify({"error": "OpenAI API key is required."}), 400
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

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

    return jsonify(result)


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    return jsonify(rag.get_stats())


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
    api_key    = data.get("api_key", "").strip()

    if not file_b64:
        return jsonify({"error": "No file data provided."}), 400

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
    if not api_key:
        return jsonify({"error": "OpenAI API key is required for image reports."}), 400

    try:
        cleaned_data_url, metrics = cv_engine.preprocess_report_image(raw_bytes)

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

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
    print("=" * 55)
    print("  MedAgent Server — http://localhost:8080")
    print("  Open index.html in your browser or visit the URL")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8080, debug=False)
