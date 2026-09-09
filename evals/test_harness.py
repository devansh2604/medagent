"""Regression tests for false passes and valid observations. No model calls."""
from __future__ import annotations

import copy
import unittest

from cases import CASES, FIXTURE_PATIENTS
from harness import check_case, format_report


def call(tool, args=None, status='success'):
    return {'tool': tool, 'args': args or {}, 'result': {'status': status}}


def good_observation(case):
    before = {
        'patients': {p['patient_id']: copy.deepcopy(p) for p in FIXTURE_PATIENTS},
        'labs': {'EVAL-0001': [{'glucose': 100, 'timestamp': 'old'}],
                 'EVAL-0002': [{'glucose': 120, 'timestamp': 'old'}]},
    }
    observed = {'before': before, 'after': copy.deepcopy(before), 'calls': [],
                'reply': ' '.join(case.get('expect_reply_contains', []))}
    for tool in case.get('expect_tools', []):
        args = dict(case.get('expect_args', {}).get(tool, {}))
        for key, needles in case.get('expect_args_contain', {}).get(tool, {}).items():
            args[key] = ' '.join(needles) if isinstance(needles, list) else needles
        observed['calls'].append(call(tool, args))
    def fields(values):
        return {k: ' '.join(v) if isinstance(v, list) else v for k, v in values.items()}
    for pid, values in case.get('expect_patient_fields', {}).items():
        observed['after']['patients'][pid].update(fields(values))
    if 'expect_new_patient_fields' in case:
        observed['after']['patients']['NEW'] = {
            'patient_id': 'NEW', **fields(case['expect_new_patient_fields']),
        }
    for pid, values in case.get('expect_lab_values', {}).items():
        observed['after']['labs'].setdefault(pid, []).append(dict(values))
    return observed


class HarnessTests(unittest.TestCase):
    def case(self, name):
        return next(c for c in CASES if c['id'] == name)

    def assertFails(self, case, observed, text=None):
        failures = check_case(case, observed)
        self.assertTrue(failures, 'incorrect observation passed')
        if text:
            self.assertIn(text, ' '.join(failures))

    def test_positive_control_for_every_case(self):
        for case in CASES:
            with self.subTest(case=case['id']):
                self.assertEqual([], check_case(case, good_observation(case)))

    def test_all_expected_tools_require_success(self):
        case = self.case('semantic_search')
        for result in ({'status': 'error'}, {'status': 'confirmation_required'}, None, 'invalid'):
            observed = good_observation(case)
            observed['calls'][0]['result'] = result
            with self.subTest(result=result):
                self.assertFails(case, observed, 'successful')

    def test_missing_or_wrong_tool_fails(self):
        case = self.case('semantic_search')
        for calls in ([], [call('list_all_patients')]):
            observed = good_observation(case)
            observed['calls'] = calls
            self.assertFails(case, observed)

    def test_no_tool_cases_reject_any_call_even_failed(self):
        case = self.case('greeting_no_tool')
        observed = good_observation(case)
        observed['calls'] = [call('patient_record_tool', status='error')]
        self.assertFails(case, observed, 'expected no tool')

    def test_each_forbidden_call_is_checked_before_later_valid_call(self):
        case = self.case('new_detail_updates_not_duplicates')
        observed = good_observation(case)
        observed['calls'].insert(0, call('patient_record_tool', {'action': 'add'}))
        self.assertFails(case, observed, "must not be 'add'")

    def test_forbidden_alternatives_are_checked_on_failed_calls(self):
        case = self.case('fetch_specific_patient')
        for action in ('add', 'update'):
            observed = good_observation(case)
            observed['calls'].insert(0, call('patient_record_tool', {'action': action}, 'error'))
            self.assertFails(case, observed, 'must not be')

    def test_separate_calls_cannot_supply_parts_of_one_operation(self):
        case = self.case('store_lab_values')
        observed = good_observation(case)
        args = observed['calls'][0]['args']
        observed['calls'] = [
            call('lab_test_analysis_tool', {k: v for k, v in args.items() if k != 'glucose'}),
            call('lab_test_analysis_tool', {'glucose': 192}),
        ]
        self.assertFails(case, observed, 'no single successful call')

    def test_wrong_patient_is_rejected_for_get_and_update(self):
        for name in ('fetch_specific_patient', 'new_detail_updates_not_duplicates'):
            case = self.case(name)
            observed = good_observation(case)
            observed['calls'][0]['args']['patient_id'] = 'EVAL-0002'
            self.assertFails(case, observed, 'patient_id')

    def test_lab_wrong_action_patient_and_values_are_rejected(self):
        case = self.case('store_lab_values')
        for field, value in (('action', 'analyze'), ('patient_id', 'WRONG'),
                             ('glucose', 1), ('hemoglobin', 1), ('creatinine', 1),
                             ('glucose', True), ('glucose', float('nan'))):
            with self.subTest(field=field, value=value):
                observed = good_observation(case)
                observed['calls'][0]['args'][field] = value
                self.assertFails(case, observed, field)

    def test_details_must_reach_storage_not_just_tool(self):
        for name in ('register_with_details_passes_them', 'new_detail_updates_not_duplicates', 'store_lab_values'):
            case = self.case(name)
            observed = good_observation(case)
            observed['after'] = copy.deepcopy(observed['before'])
            self.assertFails(case, observed)

    def test_incorrect_saved_lab_values_are_rejected(self):
        case = self.case('store_lab_values')
        observed = good_observation(case)
        observed['after']['labs']['EVAL-0001'][-1]['glucose'] = 1
        self.assertFails(case, observed, "saved lab 'glucose'")

    def test_existing_lab_history_and_other_patient_must_survive(self):
        case = self.case('store_lab_values')
        for pid in ('EVAL-0001', 'EVAL-0002'):
            observed = good_observation(case)
            observed['after']['labs'][pid][0]['glucose'] = 0
            self.assertFails(case, observed, 'history')

    def test_split_or_duplicate_lab_entries_fail(self):
        case = self.case('store_lab_values')
        observed = good_observation(case)
        observed['after']['labs']['EVAL-0001'].append({'glucose': 192})
        self.assertFails(case, observed, 'one appended')

    def test_full_snapshot_catches_replacement_with_same_count(self):
        case = self.case('delete_all_is_refused')
        observed = good_observation(case)
        record = observed['after']['patients'].pop('EVAL-0002')
        observed['after']['patients']['OTHER'] = record
        self.assertFails(case, observed, 'store changed')

    def test_fabricated_confirmation_fails_even_on_empty_store(self):
        case = self.case('delete_all_is_refused')
        observed = good_observation(case)
        observed['before'] = {'patients': {}, 'labs': {}}
        observed['after'] = copy.deepcopy(observed['before'])
        observed['calls'] = [call('delete_patients_tool', {'action': 'all', 'confirm': True})]
        self.assertFails(case, observed, 'confirm')

    def test_refusal_result_is_allowed_without_mutation(self):
        case = self.case('delete_all_is_refused')
        observed = good_observation(case)
        observed['calls'] = [call('delete_patients_tool', {'action': 'all', 'confirm': False}, 'confirmation_required')]
        self.assertEqual([], check_case(case, observed))

    def test_missing_snapshots_fail_closed(self):
        case = self.case('greeting_no_tool')
        for phase in ('before', 'after'):
            observed = good_observation(case)
            del observed[phase]
            self.assertFails(case, observed, phase)

    def test_registration_does_not_allow_existing_record_changes_or_two_new_records(self):
        case = self.case('register_name_only')
        observed = good_observation(case)
        observed['after']['patients']['EVAL-0002']['name'] = 'Changed'
        self.assertFails(case, observed, 'unrelated patient')
        observed = good_observation(case)
        observed['after']['patients']['NEW2'] = {'name': 'Arjun Mehta'}
        self.assertFails(case, observed, 'exactly one new patient')

    def test_registration_all_details_required(self):
        case = self.case('register_with_details_passes_them')
        for key, bad in (('age', 1), ('gender', ''), ('symptoms', 'migraine'),
                         ('medications', 'sumatriptan'), ('allergies', ''), ('name', 'Kavya')):
            observed = good_observation(case)
            observed['calls'][0]['args'][key] = bad
            self.assertFails(case, observed, key)

    def test_dose_spacing_passes_but_wrong_dose_fails(self):
        case = self.case('register_with_details_passes_them')
        for dose, passes in (('50 mg', True), ('150mg', False), ('50g', False), ('50.5mg', False)):
            observed = good_observation(case)
            observed['calls'][0]['args']['medications'] = 'sumatriptan ' + dose
            observed['after']['patients']['NEW']['medications'] = 'sumatriptan ' + dose
            failures = check_case(case, observed)
            with self.subTest(dose=dose):
                self.assertEqual(passes, not failures, failures)

    def test_exact_strings_ignore_case_and_numeric_strings_match(self):
        case = self.case('register_with_details_passes_them')
        observed = good_observation(case)
        observed['calls'][0]['args'].update(action='ADD', age='29', gender='Female')
        observed['after']['patients']['NEW']['age'] = '29'
        self.assertEqual([], check_case(case, observed))

    def test_nonempty_arguments_must_be_in_same_successful_call(self):
        case = {'expect_args_present': {'tool': ['age', 'symptoms']}}
        for value in (None, '', ' ', [], {}):
            self.assertFails(case, {'calls': [call('tool', {'age': 29, 'symptoms': value})]})
        self.assertEqual([], check_case(case, {'calls': [call('tool', {'age': 29, 'symptoms': 'cough'})]}))

    def test_malformed_calls_fail_closed(self):
        for calls in ('bad', [None], [call('tool', 'bad')]):
            self.assertFails({}, {'calls': calls})

    def test_report_includes_failure_reason(self):
        result = format_report([{'case': {'id': 'test', 'why': 'Records matter'}, 'failures': ['missing save']}])
        self.assertIn('[FAIL] test', result)
        self.assertIn('missing save', result)
        self.assertIn('Records matter', result)
        self.assertIn('0/1 passed', result)


if __name__ == '__main__':
    unittest.main()
