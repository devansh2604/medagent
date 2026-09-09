"""Exercise runner isolation and failure reporting without model or DB dependencies."""
from __future__ import annotations

import builtins
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import patch

import run_evals as runner


class ObserveCaseTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name).resolve()
        self.chroma = self.root / 'chroma'
        self.chroma.mkdir()
        self.labs = self.root / 'labs.json'
        self.persisted = {}
        self.agent = types.ModuleType('agent')
        self.agent._LAB_STORE_PATH = str(self.labs)
        self.agent.PATIENT_STORE = {}
        self.agent.LAB_STORE = {}
        self.agent._execute_tool = lambda name, args: json.dumps({'status': 'success'})
        self.agent.run_agent = lambda **kwargs: {'reply': 'Done'}
        self.rag = types.ModuleType('rag')
        self.rag.DB_PATH = str(self.chroma)
        self.rag.upsert_patient = lambda pid, record: self.persisted.update({pid: copy.deepcopy(record)})
        self.rag.get_all_patients = lambda: list(self.persisted.values())
        modules = patch.dict(sys.modules, agent=self.agent, rag=self.rag)
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(os.environ, {
            'MEDAGENT_CHROMA_PATH': str(self.chroma),
            'MEDAGENT_LAB_STORE': str(self.labs),
            'OPENAI_API_KEY': 'offline-test-key',
        })
        environment.start()
        self.addCleanup(environment.stop)

    def observe(self, **expectations):
        return runner._observe_case({'id': 'probe', 'message': 'test', **expectations}, self.root)

    def test_error_result_cannot_pass_expected_tool(self):
        self.agent._execute_tool = lambda name, args: json.dumps({
            'status': 'error', 'message': 'Patient not found',
        })

        def run(**kwargs):
            self.agent._execute_tool('patient_record_tool', {'action': 'update'})
            return {'reply': 'Saved successfully'}

        self.agent.run_agent = run
        result = self.observe(expect_tools=['patient_record_tool'])
        self.assertTrue(result['failures'])
        self.assertIn('successful', ' '.join(result['failures']))

    def test_in_memory_updates_do_not_count_as_persisted_patient_or_labs(self):
        def run(**kwargs):
            self.agent.PATIENT_STORE['EVAL-0001']['allergies'] = 'sulfa'
            self.agent.LAB_STORE['EVAL-0001'] = [{'glucose': 192}]
            return {'reply': 'Saved'}

        self.agent.run_agent = run
        result = self.observe(
            expect_patient_fields={'EVAL-0001': {'allergies': ['sulfa']}},
            expect_lab_values={'EVAL-0001': {'glucose': 192}},
        )
        failures = ' '.join(result['failures'])
        self.assertIn("persisted 'allergies'", failures)
        self.assertIn('appended lab entry', failures)

    def test_persisted_snapshots_are_independent_and_detect_writes(self):
        def run(**kwargs):
            # Mutate an object previously returned by get_all_patients. The
            # before snapshot must remain unchanged, not alias this object.
            self.persisted['EVAL-0001']['allergies'] = 'sulfa'
            self.labs.write_text(json.dumps({'EVAL-0001': [{'glucose': 192}]}))
            return {'reply': 'Saved'}

        self.agent.run_agent = run
        result = self.observe(expect_store_unchanged=True)
        self.assertIn('store changed', ' '.join(result['failures']))

    def test_persisted_changes_pass_even_when_cache_is_stale(self):
        def run(**kwargs):
            self.persisted['EVAL-0001']['allergies'] = 'sulfa'
            self.labs.write_text(json.dumps({'EVAL-0001': [{'glucose': 192}]}))
            return {'reply': 'Saved'}

        self.agent.run_agent = run
        result = self.observe(
            expect_patient_fields={'EVAL-0001': {'allergies': ['sulfa']}},
            expect_lab_values={'EVAL-0001': {'glucose': 192}},
        )
        self.assertEqual([], result['failures'])
        self.assertEqual('None', self.agent.PATIENT_STORE['EVAL-0001']['allergies'])
        self.assertEqual({}, self.agent.LAB_STORE)

    def test_recording_preserves_arguments_before_executor_mutates_them(self):
        def execute(name, args):
            args['action'] = 'get'
            return json.dumps({'status': 'success'})

        def run(**kwargs):
            self.agent._execute_tool('patient_record_tool', {'action': 'update'})
            return {'reply': 'Saved'}

        self.agent._execute_tool = execute
        self.agent.run_agent = run
        result = self.observe(expect_args={'patient_record_tool': {'action': 'update'}})
        self.assertEqual([], result['failures'])
        self.assertIs(execute, self.agent._execute_tool)

    def test_executor_restored_after_run_or_tool_exception(self):
        for tool_raises in (False, True):
            with self.subTest(tool_raises=tool_raises):
                def execute(name, args):
                    raise RuntimeError('tool failed')

                def run(**kwargs):
                    if tool_raises:
                        self.agent._execute_tool('patient_record_tool', {'action': 'get'})
                    raise RuntimeError('model failed')

                self.agent._execute_tool = execute
                self.agent.run_agent = run
                result = self.observe()
                self.assertIn('run_agent raised: RuntimeError', ' '.join(result['failures']))
                self.assertIs(execute, self.agent._execute_tool)

    def test_both_environment_paths_are_checked_before_store_imports(self):
        original_import = builtins.__import__
        for key in ('MEDAGENT_CHROMA_PATH', 'MEDAGENT_LAB_STORE'):
            for value in ('', str(self.root.parent / (self.root.name + '-escape'))):
                with self.subTest(key=key, value=value):
                    with patch.dict(os.environ, {key: value}):
                        with patch('builtins.__import__', wraps=original_import) as imports:
                            with self.assertRaises(RuntimeError):
                                self.observe()
                        imported_names = [call.args[0] for call in imports.call_args_list]
                        self.assertNotIn('agent', imported_names)
                        self.assertNotIn('rag', imported_names)

    def test_imported_store_paths_are_also_checked(self):
        for module, attribute in ((self.rag, 'DB_PATH'), (self.agent, '_LAB_STORE_PATH')):
            with self.subTest(attribute=attribute):
                with patch.object(module, attribute, str(self.root.parent / 'outside')):
                    with self.assertRaisesRegex(RuntimeError, 'not inside scratch'):
                        self.observe()
                self.assertEqual({}, self.persisted)

    def test_root_prefix_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            link = self.root / 'escape'
            link.symlink_to(outside, target_is_directory=True)
            for path in (self.root, self.root.parent / (self.root.name + '-sibling'), link / 'labs.json'):
                with self.subTest(path=path):
                    with self.assertRaisesRegex(RuntimeError, 'not inside scratch'):
                        runner._assert_isolated(str(path), self.root)
        runner._assert_isolated(str(self.root / 'valid' / 'labs.json'), self.root)


class AttemptTests(unittest.TestCase):
    def test_each_attempt_restores_seed_fixtures_and_globals_in_a_fresh_process(self):
        # The child uses the real observation/worker functions but replaces
        # external dependencies with tiny stores implemented using JSON.
        worker_source = '''
            import json, os, sys, types
            from pathlib import Path
            sys.path.insert(0, RUNNER_DIRECTORY)
            import run_evals as runner
            store = Path(os.environ['MEDAGENT_CHROMA_PATH']) / 'patients.json'
            labs = Path(os.environ['MEDAGENT_LAB_STORE'])
            agent = types.ModuleType('agent')
            rag = types.ModuleType('rag')
            agent._LAB_STORE_PATH = str(labs)
            agent.PATIENT_STORE = {}
            agent.LAB_STORE = {}
            agent._execute_tool = lambda name, args: '{}'
            rag.DB_PATH = str(store.parent)
            def upsert(pid, record):
                records = json.loads(store.read_text())
                records[pid] = record
                store.write_text(json.dumps(records))
            rag.upsert_patient = upsert
            rag.get_all_patients = lambda: list(json.loads(store.read_text()).values())
            def run(**kwargs):
                records = json.loads(store.read_text())
                assert records['SEEDED']['name'] == 'Original seed'
                assert agent.PATIENT_STORE == {
                    p['patient_id']: p for p in runner.FIXTURE_PATIENTS
                }
                assert agent.LAB_STORE == {}
                assert not labs.exists()
                # Every attempt deliberately corrupts all local state.
                store.write_text('{}')
                agent.PATIENT_STORE.clear()
                agent.LAB_STORE['OLD'] = [{'glucose': 1}]
                labs.write_text(json.dumps(agent.LAB_STORE))
                return {'reply': 'done'}
            agent.run_agent = run
            sys.modules.update(agent=agent, rag=rag)
            request = Path(sys.argv[2])
            code = runner._worker(request)
            output = request.parent / 'result.json'
            result = json.loads(output.read_text())
            result.update(pid=os.getpid(), scratch=str(request.parent))
            output.write_text(json.dumps(result))
            sys.exit(code)
        '''
        worker_source = textwrap.dedent(worker_source).replace('RUNNER_DIRECTORY', repr(str(runner.HERE)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / 'chroma_seed'
            seed.mkdir()
            initial = {'SEEDED': {'patient_id': 'SEEDED', 'name': 'Original seed'}}
            seed_file = seed / 'patients.json'
            seed_file.write_text(json.dumps(initial))
            worker = root / 'stub_worker.py'
            worker.write_text(worker_source)
            with patch.object(runner, 'ROOT', root), patch.object(runner, '__file__', str(worker)):
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-key'}):
                    results = [runner.run_attempt({'id': 'isolation', 'message': 'test'}, 10)
                               for _ in range(3)]
            self.assertEqual(initial, json.loads(seed_file.read_text()))
        for result in results:
            self.assertEqual([], result['failures'])
            self.assertNotEqual(os.getpid(), result['pid'])
            self.assertFalse(Path(result['scratch']).exists())
        self.assertEqual(3, len({result['pid'] for result in results}))
        self.assertEqual(3, len({result['scratch'] for result in results}))

    def test_cleanup_on_timeout_failed_worker_missing_or_invalid_result(self):
        for outcome in ('timeout', 'failed', 'missing', 'invalid', 'setup_error', 'oserror'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as seed_root:
                roots = []

                def dispatch(command, **kwargs):
                    root = Path(command[-1]).parent
                    roots.append(root)
                    if outcome == 'timeout':
                        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
                    if outcome == 'oserror':
                        raise OSError('unable to start worker')
                    if outcome == 'invalid':
                        (root / 'result.json').write_text('not json')
                    if outcome == 'setup_error':
                        (root / 'result.json').write_text(json.dumps({
                            'case': {}, 'failures': ['setup failed'], 'setup_error': True,
                        }))
                    return subprocess.CompletedProcess(command, 1 if outcome == 'failed' else 0)

                with patch.object(runner, 'ROOT', Path(seed_root)):
                    with patch.object(runner.subprocess, 'run', side_effect=dispatch):
                        result = runner.run_attempt({'id': 'cleanup', 'message': 'test'}, 1)
                        self.assertTrue(result['failures'])
                self.assertEqual(1, len(roots))
                self.assertFalse(roots[0].exists())


class CommandLineTests(unittest.TestCase):
    def test_invalid_repeat_unknown_case_and_missing_key_do_no_work(self):
        scenarios = [
            (['--repeat', '0'], 'offline-test-key'),
            (['--repeat', '-1'], 'offline-test-key'),
            (['--repeat', 'nope'], 'offline-test-key'),
            (['--only', 'does-not-exist'], 'offline-test-key'),
            ([], ''),
            ([], '   '),
        ]
        for argv, key in scenarios:
            with self.subTest(argv=argv, key=key):
                with patch.dict(os.environ, {'OPENAI_API_KEY': key}):
                    with patch.object(runner, 'run_attempt') as attempt:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            try:
                                code = runner.main(argv)
                            except SystemExit as exc:
                                code = exc.code
                        self.assertEqual(2, code)
                        attempt.assert_not_called()

    def test_repeats_dispatch_independent_attempts_and_report_failures(self):
        case = {'id': 'repeat-probe', 'message': 'test'}
        outcomes = [
            {'case': case, 'failures': []},
            {'case': case, 'failures': ['bad tool arguments']},
            {'case': case, 'failures': []},
        ]
        output = io.StringIO()
        with patch.object(runner, 'CASES', [case]):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-key'}):
                with patch.object(runner, 'run_attempt', side_effect=outcomes) as attempt:
                    with redirect_stdout(output):
                        code = runner.main(['--repeat', '3', '--timeout', '7'])
        self.assertEqual(1, code)
        self.assertEqual(3, attempt.call_count)
        self.assertTrue(all(call.args == (case, 7) for call in attempt.call_args_list))
        self.assertIn('[FAIL] repeat-probe (run 2)', output.getvalue())
        self.assertIn('2/3 passed', output.getvalue())
        self.assertEqual('repeat-probe', case['id'])


if __name__ == '__main__':
    unittest.main()
