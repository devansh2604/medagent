"""
rag.py — ChromaDB vector database operations for MedAgent
Uses ChromaDB's built-in sentence-transformers embeddings (no OpenAI key needed)
"""
from __future__ import annotations

import chromadb
from collections import Counter

# ── ChromaDB client & collection ──────────────────────────────────────────────
_client = chromadb.PersistentClient(path="./chroma")

_collection = _client.get_or_create_collection(
    name="patient_records",
    metadata={"hnsw:space": "cosine"},
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def patient_to_text(record: dict) -> str:
    """Convert a patient dict to natural-language text for embedding."""
    return (
        f"Patient Name: {record.get('name', 'Unknown')}\n"
        f"Age: {record.get('age', 'N/A')} | Gender: {record.get('gender', 'N/A')}\n"
        f"Symptoms: {record.get('symptoms', 'None')}\n"
        f"Medical History: {record.get('medical_history', 'None')}\n"
        f"Medications: {record.get('medications', 'None')}\n"
        f"Allergies: {record.get('allergies', 'None')}\n"
        f"Diagnosis: {record.get('diagnosis', 'Pending')}"
    )


def _parse_text_fields(text: str) -> dict:
    """
    Extract structured fields from a patient text blob.
    Used as a fallback for legacy records that predate the full-metadata schema.
    """
    out = {}
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith("Patient Name:"):
            out["name"] = line[len("Patient Name:"):].strip()
        elif line.startswith("Age:") and "|" in line:
            parts = line.split("|")
            out["age"]    = parts[0].replace("Age:", "").strip()
            out["gender"] = parts[1].replace("Gender:", "").strip()
        elif line.startswith("Symptoms:"):
            out["symptoms"] = line[len("Symptoms:"):].strip()
        elif line.startswith("Medical History:"):
            out["medical_history"] = line[len("Medical History:"):].strip()
        elif line.startswith("Medications:"):
            out["medications"] = line[len("Medications:"):].strip()
        elif line.startswith("Allergies:"):
            out["allergies"] = line[len("Allergies:"):].strip()
        elif line.startswith("Diagnosis:"):
            out["diagnosis"] = line[len("Diagnosis:"):].strip()
    return out


def _build_record(metadata: dict, text: str) -> dict:
    """
    Build a structured record dict from ChromaDB metadata + document text.
    New records: all fields are in metadata.
    Legacy records: only patient_id + name in metadata; rest is parsed from text.
    """
    # Only fall back to text parsing if the key clinical fields are absent
    parsed = _parse_text_fields(text) if not metadata.get("diagnosis") else {}
    return {
        "patient_id":      metadata.get("patient_id", ""),
        "name":            metadata.get("name")            or parsed.get("name",            "Unknown"),
        "age":             metadata.get("age")             or parsed.get("age",             "N/A"),
        "gender":          metadata.get("gender")          or parsed.get("gender",          "N/A"),
        "diagnosis":       metadata.get("diagnosis")       or parsed.get("diagnosis",       "Pending"),
        "symptoms":        metadata.get("symptoms")        or parsed.get("symptoms",        "None"),
        "medications":     metadata.get("medications")     or parsed.get("medications",     "None"),
        "allergies":       metadata.get("allergies")       or parsed.get("allergies",       "None"),
        "medical_history": metadata.get("medical_history") or parsed.get("medical_history", "None"),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_patient(patient_id: str, record: dict) -> None:
    """Embed and upsert a patient record into ChromaDB with full metadata."""
    text = patient_to_text(record)
    _collection.upsert(
        ids=[patient_id],
        documents=[text],
        metadatas=[{
            "patient_id":      patient_id,
            "name":            str(record.get("name",            "")),
            "age":             str(record.get("age",             "")),
            "gender":          str(record.get("gender",          "")),
            "diagnosis":       str(record.get("diagnosis",       "")),
            "symptoms":        str(record.get("symptoms",        "")),
            "medications":     str(record.get("medications",     "")),
            "allergies":       str(record.get("allergies",       "")),
            "medical_history": str(record.get("medical_history", "")),
        }],
    )


def search_patients(query: str, n_results: int = 20) -> list[dict]:
    """
    Semantic search over patient records.
    Returns structured patient records ranked by similarity.
    """
    count = _collection.count()
    if count == 0:
        return []

    n = min(n_results, count)
    results = _collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        record = _build_record(results["metadatas"][0][i], results["documents"][0][i])
        record["similarity"] = round(1 - results["distances"][0][i], 4)
        output.append(record)
    return output


def get_all_patients() -> list[dict]:
    """Return every patient record in the collection (no query needed)."""
    count = _collection.count()
    if count == 0:
        return []
    results = _collection.get(include=["documents", "metadatas"])
    output = []
    for i in range(len(results["ids"])):
        output.append(_build_record(results["metadatas"][i], results["documents"][i]))
    return output


def get_patient(patient_id: str) -> dict | None:
    """Fetch a single patient's full structured record by exact ID."""
    results = _collection.get(ids=[patient_id], include=["documents", "metadatas"])
    if not results["ids"]:
        return None
    return _build_record(results["metadatas"][0], results["documents"][0])


def get_aggregate_stats() -> dict:
    """
    Return aggregate stats across all patients:
    count by diagnosis, gender distribution, age range.
    Works for both new records (full metadata) and legacy records (text-parsed).
    """
    results = _collection.get(include=["documents", "metadatas"])
    if not results["ids"]:
        return {"total_patients": 0}

    diagnoses = Counter()
    genders   = Counter()
    ages      = []

    for i, m in enumerate(results["metadatas"]):
        record = _build_record(m, results["documents"][i])

        dx = record.get("diagnosis") or "Unknown"
        diagnoses[dx] += 1

        g = record.get("gender") or ""
        if g and g != "N/A":
            genders[g] += 1

        try:
            ages.append(int(record.get("age", "")))
        except (ValueError, TypeError):
            pass

    return {
        "total_patients": len(results["ids"]),
        "by_diagnosis":   dict(diagnoses.most_common()),
        "by_gender":      dict(genders),
        "average_age":    round(sum(ages) / len(ages), 1) if ages else None,
        "age_range":      {"min": min(ages), "max": max(ages)} if ages else None,
    }


def delete_patient(patient_id: str) -> bool:
    """Delete a single patient record by ID. Returns True if it existed."""
    existing = _collection.get(ids=[patient_id])
    if not existing["ids"]:
        return False
    _collection.delete(ids=[patient_id])
    return True


def delete_all_patients() -> int:
    """Delete every record in the collection. Returns the count of deleted records."""
    count = _collection.count()
    if count == 0:
        return 0
    all_ids = _collection.get(include=[])["ids"]
    _collection.delete(ids=all_ids)
    return count


def get_stats() -> dict:
    """Return basic stats about the vector database."""
    count = _collection.count()
    return {
        "total_records":    count,
        "vector_dims":      384,
        "model":            "chromadb-default",
        "similarity_metric": "cosine",
    }
