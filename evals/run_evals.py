"""Run each live eval in a separate process with fresh, disposable stores.

    OPENAI_API_KEY=... python3 evals/run_evals.py --repeat 3
    python3 -m unittest discover -s evals -p 'test_*.py'
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from harness import check_case, format_report
from cases import CASES, FIXTURE_PATIENTS


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def _assert_isolated(path: str, root: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f'store is not inside scratch directory: {resolved}')


def _observe_case(case: dict, root: Path) -> dict:
    # Check environment paths before imports, which open the stores.
    root = root.resolve()
    for key in ('MEDAGENT_CHROMA_PATH', 'MEDAGENT_LAB_STORE'):
        if not os.environ.get(key):
            raise RuntimeError(f'{key} is not set')
        _assert_isolated(os.environ[key], root)

    import agent
    import rag

    _assert_isolated(rag.DB_PATH, root)
    _assert_isolated(agent._LAB_STORE_PATH, root)
    for fixture in FIXTURE_PATIENTS:
        rag.upsert_patient(fixture['patient_id'], dict(fixture))
        agent.PATIENT_STORE[fixture['patient_id']] = dict(fixture)

    def snapshot() -> dict:
        # Read persisted data, not the agent response or in-memory caches.
        patients = {p['patient_id']: p for p in rag.get_all_patients()}
        lab_path = Path(agent._LAB_STORE_PATH)
        labs = json.loads(lab_path.read_text()) if lab_path.exists() else {}
        return copy.deepcopy({'patients': patients, 'labs': labs})

    calls = []
    real_execute = agent._execute_tool

    def recording_execute(name, tool_args):
        call = {'tool': name, 'args': copy.deepcopy(tool_args), 'result': None}
        calls.append(call)
        try:
            result = real_execute(name, tool_args)
        except Exception as exc:
            call['result'] = {'status': 'error', 'message': str(exc)}
            raise
        try:
            call['result'] = json.loads(result)
        except (TypeError, ValueError):
            call['result'] = result
        return result

    before = snapshot()
    agent._execute_tool = recording_execute
    reply = ''
    run_error = None
    try:
        out = agent.run_agent(
            api_key=os.environ['OPENAI_API_KEY'],
            patient_id=case.get('patient_id', ''),
            message=case['message'],
            history=[],
        )
        reply = out.get('reply', '')
    except Exception as exc:
        run_error = f'run_agent raised: {type(exc).__name__}: {exc}'
    finally:
        agent._execute_tool = real_execute

    observed = {'calls': calls, 'reply': reply, 'before': before, 'after': snapshot()}
    failures = check_case(case, observed)
    if run_error:
        failures.append(run_error)
    return {'case': case, 'failures': failures}


def _worker(request_path: Path) -> int:
    request = json.loads(request_path.read_text())
    response_path = request_path.parent / 'result.json'
    try:
        result = _observe_case(request['case'], request_path.parent)
    except Exception as exc:
        result = {'case': request['case'], 'failures': [
            f'worker setup failed: {type(exc).__name__}: {exc}'], 'setup_error': True}
    response_path.write_text(json.dumps(result, indent=2))
    return 0


def run_attempt(case: dict, timeout: int) -> dict:
    # Fresh processes avoid Chroma client caches and agent globals carrying
    # mutations into the next case or repeat.
    with tempfile.TemporaryDirectory(prefix='medagent-evals-') as directory:
        root = Path(directory)
        chroma = root / 'chroma'
        seed = ROOT / 'chroma_seed'
        if seed.is_dir():
            shutil.copytree(seed, chroma)
        else:
            chroma.mkdir()
        env = dict(os.environ)
        env.update(MEDAGENT_CHROMA_PATH=str(chroma),
                   MEDAGENT_LAB_STORE=str(root / 'lab_store.json'),
                   DEMO_MODE='false', PYTHONDONTWRITEBYTECODE='1')
        request = root / 'request.json'
        request.write_text(json.dumps({'case': case}))
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--_worker', str(request)],
                env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {'case': case, 'failures': [f'worker timed out after {timeout}s']}
        except OSError as exc:
            return {'case': case, 'failures': [f'could not start worker: {exc}'],
                    'setup_error': True}
        response = root / 'result.json'
        if completed.returncode or not response.exists():
            return {'case': case, 'failures': [
                f'worker exited {completed.returncode} without a valid result'],
                'setup_error': True}
        try:
            return json.loads(response.read_text())
        except (OSError, ValueError) as exc:
            return {'case': case, 'failures': [f'invalid worker result: {exc}'],
                    'setup_error': True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', help='run one case by id')
    ap.add_argument('--repeat', type=_positive_int, default=1)
    ap.add_argument('--timeout', type=_positive_int, default=300,
                    help='maximum seconds per attempt (default: 300)')
    ap.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args._worker:
        return _worker(args._worker)
    selected = [c for c in CASES if not args.only or c['id'] == args.only]
    if not selected:
        print(f'no case matching {args.only!r}')
        return 2
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        print('OPENAI_API_KEY is not set. These cases call the real model.')
        return 2
    results = []
    for case in selected:
        for attempt in range(args.repeat):
            label = dict(case)
            if args.repeat > 1:
                label['id'] = f"{case['id']} (run {attempt + 1})"
            print(f"Running {label['id']}...", flush=True)
            try:
                result = run_attempt(case, args.timeout)
            except OSError as exc:
                result = {'case': case, 'failures': [f'could not prepare scratch store: {exc}'],
                          'setup_error': True}
            result['case'] = label
            results.append(result)
    print(format_report(results))
    if any(r.get('setup_error') for r in results):
        return 2
    return 1 if any(r['failures'] for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
