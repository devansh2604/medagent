# Architecture notes

Context for working on MedAgent: the commands, and the decisions that are
expensive to rediscover by reading the code.

## Commands

There is no build step, linter, or test suite. The venv is Python 3.9 and is not on PATH.

```bash
venv/bin/python server.py          # run locally on :8080 (own-key mode)
venv/bin/pip install -r requirements.txt
venv/bin/python seed_database.py   # generate synthetic patients into ChromaDB
```

Restart the server after any `.py` change — nothing hot-reloads. `index.html` only needs a browser refresh.

```bash
kill $(lsof -ti :8080); venv/bin/python server.py
```

Run as the container host does (gunicorn, demo mode, port 7860):

```bash
DEMO_MODE=true OPENAI_API_KEY=sk-... venv/bin/gunicorn server:app \
  --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 180
```

## Verifying changes

Endpoint tests pass while the agent is broken. Three bugs shipped that way: the
session Patient ID was injected into `action='add'` (filing patients under junk
ids, and silently overwriting existing records because storage is an upsert),
the model called tools without passing the details it then claimed to have
saved, and it had no notion of the active patient so it created duplicates
instead of updating.

Exercise agent behaviour through the running UI with a live model, and say
plainly what was not covered.

## Architecture

`index.html` (single page, no framework) → Flask (`server.py`) → `run_agent()`
(`agent.py`) → OpenAI with `tool_choice="auto"` → tools execute in-process →
ChromaDB (`rag.py`) and JSON files.

**The agent loop** lives in `run_agent()`: up to 15 iterations of
completion → execute any tool calls → feed results back, exiting when the model
returns no tool call. It returns `reply`, `patient_record`, `lab_results` and
`tools_called`; the frontend drives the whole dashboard off that one payload.

**The active patient is injected into the system prompt every turn** from
`PATIENT_STORE`/ChromaDB. Without it the model cannot tell an existing patient
from a new one and creates duplicates. `run_agent` also injects the session
`patient_id` into tool args — but deliberately **not** into
`patient_record_tool(action='add')`, and `add` refuses to overwrite an id that
already exists.

**Which patient the dashboard shows** is decided by `touched_pids`: the ids
actually read or written this turn, last one wins, falling back to the session
id.

**State lives in three places.** `PATIENT_STORE` (in-memory, mirrored to
ChromaDB, so it survives restarts), `LAB_STORE` → `lab_store.json`, and
`SESSION_HISTORY` → `session_history.json` (gitignored). `rag.py` anchors the
ChromaDB path to the module, not the working directory.

**`chroma_seed/` is a prebuilt database committed to the repo.** `server.py`
copies it into `chroma/` at boot *before* importing `rag`, when `chroma/` is
empty. This exists because regenerating 90 embeddings on a small instance
saturates the CPU for minutes and starves the web workers — the service answers
one request then goes dark. The shipped database always wins over
`SEED_ON_BOOT`, which hosts often leave set from an earlier deploy.

**Two modes, local by default.** `DEMO_MODE=true` switches to a server-side key,
strips `DESTRUCTIVE_TOOLS` from the tool list *and* refuses them again in
`_execute_tool`, and rate limits per IP. The rate limiter is in-memory, so run a
single worker.

**Guards belong in the executor, not the prompt.** `delete_patients_tool`
with `action='all'` returns `confirmation_required` unless `confirm=true` is
passed; prompt instructions alone were not enough.

**Report analyser** (`/api/analyze-report`): images go through `cv_engine.py`
(CLAHE, adaptive threshold, deskew) then GPT-4o Vision; PDFs go through pypdf.
Extracted text is fed back into the chat agent. `cv_engine` is imported lazily
inside the route — OpenCV costs ~31MB resident and is only needed for images.

**Lab classification** is a rule table, `LAB_NORMAL_RANGES` in `agent.py`, applied
by `_classify_value`.

## Deployment

`render.yaml` and `Procfile` target Render; `Dockerfile` targets Hugging Face
Spaces (port 7860, UID 1000, embedding model baked in at build). Free Render
could not serve this — 0.1 CPU cannot load the ONNX embedding model and answer
requests at the same time. Free HF Spaces now requires a paid plan for Docker.

`OPENAI_API_KEY` is never committed; it is set in the host dashboard.
