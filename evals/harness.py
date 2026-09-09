"""
Assertion logic for the eval suite.

Deliberately separate from the runner and free of any OpenAI or ChromaDB
import, so the checker itself can be tested offline — a suite you cannot
trust is worse than no suite.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


def _equal_value(got, want) -> bool:
    """Compare numeric values without rounding or treating booleans as numbers."""
    if isinstance(got, bool) or isinstance(want, bool):
        return type(got) is type(want) and got == want
    if isinstance(got, (int, float, str)) and isinstance(want, (int, float, str)):
        try:
            actual, expected = Decimal(str(got).strip()), Decimal(str(want).strip())
        except InvalidOperation:
            pass
        else:
            return actual.is_finite() and expected.is_finite() and actual == expected
    if isinstance(got, str) and isinstance(want, str):
        return got.strip().casefold() == want.strip().casefold()
    return type(got) is type(want) and got == want


def _contains_all(got, needles) -> bool:
    if got is None:
        return False
    if not isinstance(needles, list):
        needles = [needles]
    actual = str(got).casefold()
    for needle in needles:
        needle = str(needle).casefold()
        dose = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", needle)
        if dose:
            # The same dose may be written as '50mg' or '50 mg', but 150mg
            # must not pass simply because it contains the text '50mg'.
            amount, unit = dose.groups()
            pattern = rf"(?<![\d.]){re.escape(amount)}\s*{re.escape(unit)}(?!\w)"
            if not re.search(pattern, actual):
                return False
        elif needle not in actual:
            return False
    return True


def _field_matches(got, want) -> bool:
    # Persisted free text may be paraphrased. A list explicitly requests all
    # substrings; scalar expectations remain exact.
    return _contains_all(got, want) if isinstance(want, list) else _equal_value(got, want)


def _same_state(before, after) -> bool:
    """Deep equality that also catches type changes such as 1 becoming True."""
    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(
            _same_state(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, list):
        return len(before) == len(after) and all(
            _same_state(left, right) for left, right in zip(before, after)
        )
    return before == after


def check_case(case: dict, observed: dict) -> list[str]:
    """
    Compare one case against what actually happened.

    observed: {
        "calls": [{"tool": "patient_record_tool", "args": {...},
                   "result": {"status": "success", ...}}, ...],
        "reply": "...",
        "before": {"patients": {patient_id: record}, "labs": {patient_id: [...]}},
        "after":  {"patients": {patient_id: record}, "labs": {patient_id: [...]}},
    }

    All argument expectations for a tool must match ONE successful call.
    Forbidden arguments are checked on EVERY call, including failed calls.
    Tool names and patient counts are derived from evidence, never summaries.
    Returns a list of failure strings. Empty means the case passed.
    """
    failures: list[str] = []
    calls = []
    raw_calls = observed.get("calls", [])
    if not isinstance(raw_calls, list):
        failures.append("calls must be a list of recorded tool calls")
        raw_calls = []
    for index, call in enumerate(raw_calls, 1):
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
            failures.append(f"call {index}: invalid recorded tool call")
            continue
        calls.append(call)
        if not isinstance(call.get("args"), dict):
            failures.append(f"call {index} ({call['tool']}): args must be a dictionary")
    tools = [call["tool"] for call in calls]
    reply = observed.get("reply", "") or ""

    def successful(call: dict) -> bool:
        result = call.get("result")
        return isinstance(result, dict) and result.get("status") == "success"

    # Expected tools. An explicit empty list means "no tool at all".
    if "expect_tools" in case:
        expected = case["expect_tools"]
        if expected == []:
            if tools:
                failures.append(f"expected no tool call, got {sorted(set(tools))}")
        else:
            for tool in expected:
                if not any(call["tool"] == tool and successful(call) for call in calls):
                    failures.append(f"expected a successful {tool} call, got {sorted(set(tools)) or 'none'}")

    for tool in case.get("forbid_tools", []):
        if tool in tools:
            failures.append(f"{tool} should not have been called")

    # Keep every constraint together so unrelated calls cannot satisfy pieces
    # of one expected operation (or hide an earlier forbidden operation).
    argument_tools = dict.fromkeys(
        tool for kind in ("expect_args", "expect_args_contain", "expect_args_present")
        for tool in case.get(kind, {})
    )
    for tool in argument_tools:
        mismatches = []
        matched = False
        for call in calls:
            if call["tool"] != tool or not successful(call):
                continue
            actual = call.get("args")
            if not isinstance(actual, dict):
                continue
            errors = []
            for key, want in case.get("expect_args", {}).get(tool, {}).items():
                if key not in actual or not _equal_value(actual[key], want):
                    errors.append(f"arg '{key}' was {actual.get(key)!r}, expected {want!r}")
            for key, needles in case.get("expect_args_contain", {}).get(tool, {}).items():
                if not _contains_all(actual.get(key), needles):
                    errors.append(f"arg '{key}' was {actual.get(key)!r}, should mention {needles!r}")
            for key in case.get("expect_args_present", {}).get(tool, []):
                value = actual.get(key)
                if value is None or (isinstance(value, (str, list, dict)) and not value):
                    errors.append(f"arg '{key}' missing or empty")
                elif isinstance(value, str) and not value.strip():
                    errors.append(f"arg '{key}' missing or empty")
            if not errors:
                matched = True
                break
            mismatches.append(errors)
        if not matched:
            detail = "; ".join(min(mismatches, key=len)) if mismatches else "no successful call with valid arguments"
            failures.append(f"{tool}: no single successful call matched all expected arguments ({detail})")

    for tool, forbidden in case.get("forbid_args", {}).items():
        for index, call in enumerate(calls, 1):
            actual = call.get("args")
            if call["tool"] != tool or not isinstance(actual, dict):
                continue
            for key, bad in forbidden.items():
                alternatives = bad if isinstance(bad, list) else [bad]
                if key in actual and any(_equal_value(actual[key], value) for value in alternatives):
                    failures.append(f"{tool} call {index}: arg '{key}' must not be {actual[key]!r}")

    for needle in case.get("expect_reply_contains", []):
        if needle.lower() not in reply.lower():
            failures.append(f"reply should contain {needle!r}")

    state_keys = (
        "expect_patient_count_delta", "expect_patient_fields", "expect_new_patient_fields",
        "expect_lab_values", "expect_store_unchanged", "expect_unchanged_other_patients",
    )
    if any(key in case for key in state_keys):
        snapshots = []
        for phase in ("before", "after"):
            snapshot = observed.get(phase)
            if not isinstance(snapshot, dict) or not all(
                isinstance(snapshot.get(store), dict) for store in ("patients", "labs")
            ):
                failures.append(f"missing valid {phase} store snapshot")
            else:
                snapshots.append(snapshot)
        if len(snapshots) != 2:
            return failures
        before, after = snapshots
        patients_before, patients_after = before["patients"], after["patients"]

        if "expect_patient_count_delta" in case:
            want = case["expect_patient_count_delta"]
            got = len(patients_after) - len(patients_before)
            if got != want:
                failures.append(f"patient count changed by {got}, expected {want}")

        def check_fields(record, expected, label):
            if not isinstance(record, dict):
                failures.append(f"{label}: persisted record missing or invalid")
                return
            for key, want in expected.items():
                if key not in record or not _field_matches(record[key], want):
                    failures.append(f"{label}: persisted '{key}' was {record.get(key)!r}, expected {want!r}")

        for pid, expected in case.get("expect_patient_fields", {}).items():
            check_fields(patients_after.get(pid), expected, f"patient {pid}")

        if "expect_new_patient_fields" in case:
            new_ids = patients_after.keys() - patients_before.keys()
            if len(new_ids) != 1:
                failures.append(f"expected exactly one new patient, got {len(new_ids)}")
            else:
                pid = next(iter(new_ids))
                check_fields(patients_after[pid], case["expect_new_patient_fields"], f"new patient {pid}")

        if "expect_unchanged_other_patients" in case:
            allowed = set(case["expect_unchanged_other_patients"])
            for pid, record in patients_before.items():
                if pid not in allowed and (
                    pid not in patients_after or not _same_state(record, patients_after[pid])
                ):
                    failures.append(f"unrelated patient {pid} was changed or deleted")

        if "expect_lab_values" in case:
            expected_labs = case["expect_lab_values"]
            for pid, expected in expected_labs.items():
                old = before["labs"].get(pid, [])
                new = after["labs"].get(pid, [])
                if not isinstance(old, list) or not isinstance(new, list):
                    failures.append(f"patient {pid}: lab history missing or invalid")
                    continue
                if len(new) != len(old) + 1 or not _same_state(old, new[:len(old)]):
                    failures.append(f"patient {pid}: expected one appended lab entry with previous history preserved")
                    continue
                entry = new[-1]
                if not isinstance(entry, dict):
                    failures.append(f"patient {pid}: appended lab entry is invalid")
                    continue
                for key, want in expected.items():
                    if key not in entry or not _equal_value(entry[key], want):
                        failures.append(f"patient {pid}: saved lab '{key}' was {entry.get(key)!r}, expected {want!r}")
            for pid in before["labs"].keys() | after["labs"].keys():
                if pid not in expected_labs and (
                    pid not in before["labs"] or pid not in after["labs"]
                    or not _same_state(before["labs"][pid], after["labs"][pid])
                ):
                    failures.append(f"unrelated patient {pid}: lab history was changed")

        if case.get("expect_store_unchanged") and not _same_state(before, after):
            failures.append("patient or lab store changed, expected both stores unchanged")

    return failures


def format_report(results: list[dict]) -> str:
    """Render results as a readable report. results: [{case, failures}]"""
    lines = []
    passed = [r for r in results if not r["failures"]]
    failed = [r for r in results if r["failures"]]

    for r in results:
        mark = "PASS" if not r["failures"] else "FAIL"
        lines.append(f"[{mark}] {r['case']['id']}")
        for f in r["failures"]:
            lines.append(f"         {f}")
        if r["failures"] and r["case"].get("why"):
            lines.append(f"         why this matters: {r['case']['why']}")

    lines.append("")
    lines.append(f"{len(passed)}/{len(results)} passed")
    if failed:
        lines.append("failed: " + ", ".join(r["case"]["id"] for r in failed))
    return "\n".join(lines)
