"""
Run the eval cases against the real agent.

    OPENAI_API_KEY=sk-... python3 evals/run_evals.py
    OPENAI_API_KEY=sk-... python3 evals/run_evals.py --only register_with_details_passes_them
    OPENAI_API_KEY=sk-... python3 evals/run_evals.py --repeat 3

Each case is a live model call, so a full run costs real money — a handful of
cents on gpt-4o-mini. It never touches the real database: a scratch ChromaDB is
built from chroma_seed/ (or empty) in a temp directory and thrown away after.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from harness import check_case, format_report   # noqa: E402
from cases import CASES, FIXTURE_PATIENT        # noqa: E402


def _isolate_stores() -> str:
    """
    Point the stores at a throwaway copy BEFORE importing agent/rag, which read
    these paths at import time. Returns the temp directory.
    """
    tmp = tempfile.mkdtemp(prefix="medagent-evals-")
    chroma = os.path.join(tmp, "chroma")

    seed = os.path.join(ROOT, "chroma_seed")
    if os.path.isdir(seed):
        shutil.copytree(seed, chroma)
    else:
        os.makedirs(chroma, exist_ok=True)

    os.environ["MEDAGENT_CHROMA_PATH"] = chroma
    os.environ["MEDAGENT_LAB_STORE"]   = os.path.join(tmp, "lab_store.json")
    os.environ.pop("DEMO_MODE", None)   # evals exercise the full tool set
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single case by id")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each case N times; the model is sampled, not deterministic")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set. These cases call the real model.")
        return 2

    tmp = _isolate_stores()
    print(f"scratch store: {tmp}\n")

    try:
        import agent, rag                      # noqa: E402  (after env is set)

        # Guard against ever pointing at the real database by mistake.
        assert rag.DB_PATH.startswith(tmp), f"not isolated! DB_PATH={rag.DB_PATH}"

        rag.upsert_patient(FIXTURE_PATIENT["patient_id"], FIXTURE_PATIENT)
        agent.PATIENT_STORE[FIXTURE_PATIENT["patient_id"]] = dict(FIXTURE_PATIENT)

        selected = [c for c in CASES if not args.only or c["id"] == args.only]
        if not selected:
            print(f"no case matching {args.only!r}")
            return 2

        # Record what the agent actually passed to each tool, without changing
        # production code: wrap the executor and log every call.
        real_execute = agent._execute_tool
        recorded: list[tuple[str, dict]] = []

        def recording_execute(name, tool_args):
            recorded.append((name, dict(tool_args)))
            return real_execute(name, tool_args)

        agent._execute_tool = recording_execute

        results = []
        for case in selected:
            for attempt in range(args.repeat):
                recorded.clear()
                before = rag.get_stats()["total_records"]
                try:
                    out = agent.run_agent(
                        api_key=api_key,
                        patient_id=case.get("patient_id", ""),
                        message=case["message"],
                        history=[],            # each case is a cold turn
                    )
                    reply = out.get("reply", "")
                except Exception as e:
                    results.append({"case": case, "failures": [f"run_agent raised: {e}"]})
                    continue

                after = rag.get_stats()["total_records"]
                observed = {
                    "tools":       [n for n, _ in recorded],
                    "calls":       list(recorded),
                    "reply":       reply,
                    "count_delta": after - before,
                }
                label = dict(case)
                if args.repeat > 1:
                    label = {**case, "id": f"{case['id']} (run {attempt + 1})"}
                results.append({"case": label, "failures": check_case(case, observed)})

        agent._execute_tool = real_execute
        print(format_report(results))
        return 1 if any(r["failures"] for r in results) else 0

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
