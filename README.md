# 🏥 MedAgent — AI Healthcare Agentic System

An agentic AI healthcare assistant powered by OpenAI GPT-4o with specialized tools.

## 🛠 Tools

| Tool | Purpose |
|------|---------|
| `collect_patient_data` | Structured intake — name, age, symptoms, vitals, medications, allergies |
| `summarize_patient` | Auto-generates a clinical summary from stored record |
| `diagnose_disease` | Infers possible conditions from symptoms and history |
| `detect_mental_health` | Analyzes chat tone for depression, anxiety, crisis signals |
| `check_medication_interactions` | Flags dangerous drug-drug interactions |
| `update_patient_record` | Dynamically updates any field as conversation progresses |

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python server.py
```

### 3. Open the UI
Visit `http://localhost:5000` in your browser.

### 4. Enter your OpenAI API Key
Paste your key in the left panel. The system uses **GPT-4o** for best results.

## 🏗 Architecture

```
User Message
    │
    ▼
Flask API (/api/chat)
    │
    ▼
run_agent() — Agentic Loop
    │
    ├── detect_mental_health()    ← Always runs on every message
    │
    ▼
OpenAI GPT-4o (function calling)
    │
    ├── collect_patient_data()
    ├── summarize_patient()
    ├── diagnose_disease()
    ├── check_medication_interactions()
    └── update_patient_record()
    │
    ▼
Final Response → UI
```

## ⚙️ Configuration

- **Model**: GPT-4o (change in `agent.py` → `run_agent()`)
- **Patient Store**: In-memory dict (replace with PostgreSQL/MongoDB for production)
- **Interaction DB**: Rule-based (replace with DrugBank API for production)
- **Diagnosis Engine**: Heuristic (replace with ICD-10 API or ML model for production)

## ⚠️ Disclaimer

This system is for **educational and demonstration purposes only**.
It is NOT a replacement for professional medical advice, diagnosis, or treatment.
Always consult a licensed healthcare professional.

## 🔐 Production Considerations

- Store patient data in encrypted database (HIPAA-compliant)
- Add authentication (OAuth2 / JWT)
- Use DrugBank or RxNorm API for real drug interactions
- Integrate ICD-10 / SNOMED for proper diagnosis coding
- Add audit logging for all tool calls
- Deploy behind HTTPS with proper CORS configuration
