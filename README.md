# 🏥 MedAgent — AI Healthcare Agentic System

An agentic AI healthcare assistant powered by OpenAI **GPT-4o-mini** with function calling, **ChromaDB** vector search (RAG) over patient records, and an **OpenCV + Vision** medical-report analyser.

## 🛠 Tools

The LLM orchestrates all tool calls itself (`tool_choice="auto"`):

| Tool | Purpose |
|------|---------|
| `patient_record_tool` | Get, add, or update patient records (in-memory + ChromaDB) |
| `rag_search_patients` | Semantic vector search across all patient records ("find diabetes patients") |
| `list_all_patients` | Fetch every patient name + ID from ChromaDB |
| `delete_patients_tool` | Delete one patient, a bulk list, or all patients |
| `lab_test_analysis_tool` | Store lab values, classify them against normal ranges, track history/trends |

## 📄 Report Analyser

Upload a medical report from the UI:

- **Image** → OpenCV preprocessing (CLAHE contrast + adaptive threshold + deskew) → GPT-4o-mini Vision extracts the text
- **PDF** → direct text extraction via pypdf

The extracted text is fed back into the chat agent, which pulls out lab values and stores them.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Seed the database with demo patients
```bash
python seed_database.py
```

### 3. Run the server
```bash
python server.py
```

### 4. Open the UI
Visit `http://localhost:8080` in your browser.

### 5. Enter your OpenAI API Key
Paste your key in the left panel. The system uses **gpt-4o-mini** (change in `agent.py` → `run_agent()`).

## 🏗 Architecture

```
User Message
    │
    ▼
Flask API (server.py — /api/chat)
    │
    ▼
run_agent() — agentic loop (agent.py)
    │
    ▼
OpenAI gpt-4o-mini (function calling, tool_choice=auto)
    │
    ├── patient_record_tool ──────┐
    ├── rag_search_patients ──────┤──► ChromaDB (rag.py, cosine / 384-dim)
    ├── list_all_patients ────────┤
    ├── delete_patients_tool ─────┘
    └── lab_test_analysis_tool ──────► lab_store.json (persisted to disk)
    │
    ▼
Final Response + patient record + classified labs → UI dashboard
```

## 📦 Project Files

| File | Role |
|------|------|
| `server.py` | Flask API server (chat, patient/lab endpoints, report analyser) |
| `agent.py` | Agentic loop, tool definitions, tool executor, lab classification |
| `rag.py` | ChromaDB operations: embed, upsert, semantic search, aggregate stats |
| `cv_engine.py` | OpenCV image preprocessing for report photos |
| `seed_database.py` | Generates demo patients into ChromaDB |
| `index.html` | Single-page UI: chat, tool status, patient dashboard |

## ⚙️ Configuration

- **Model**: gpt-4o-mini (change in `agent.py` → `run_agent()`)
- **Patient store**: in-memory dict, mirrored to ChromaDB (survives restarts via ChromaDB)
- **Lab store**: `lab_store.json`, persisted on every write
- **Normal ranges**: rule-based table in `agent.py` (`LAB_NORMAL_RANGES`)

## ⚠️ Disclaimer

This system is for **educational and demonstration purposes only**.
It is NOT a replacement for professional medical advice, diagnosis, or treatment.
Always consult a licensed healthcare professional.

## 🔐 Production Considerations

- Store patient data in an encrypted, HIPAA-compliant database
- Add authentication (OAuth2 / JWT)
- Use DrugBank or RxNorm API for real drug-interaction checking
- Integrate ICD-10 / SNOMED for proper diagnosis coding
- Add audit logging for all tool calls
- Deploy behind HTTPS with proper CORS configuration
