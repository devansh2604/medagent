"""
Assertion logic for the eval suite.

Deliberately separate from the runner and free of any OpenAI or ChromaDB
import, so the checker itself can be tested offline — a suite you cannot
trust is worse than no suite.
"""
from __future__ import annotations


def check_case(case: dict, observed: dict) -> list[str]:
    """
    Compare one case against what actually happened.

    observed: {
        "tools":        ["patient_record_tool", ...],   # names, in call order
        "calls":        [("patient_record_tool", {...}), ...],
        "reply":        "...",
        "count_delta":  int,   # change in patient count over the turn
    }

    Returns a list of failure strings. Empty means the case passed.
    """
    failures: list[str] = []
    tools = observed.get("tools", [])
    calls = observed.get("calls", [])
    reply = observed.get("reply", "") or ""

    def args_for(tool: str) -> dict:
        """Merge every call to this tool; a tool may be called more than once."""
        merged: dict = {}
        for name, args in calls:
            if name == tool and isinstance(args, dict):
                merged.update(args)
        return merged

    # Expected tools. An explicit empty list means "no tool at all".
    if "expect_tools" in case:
        expected = case["expect_tools"]
        if expected == []:
            if tools:
                failures.append(f"expected no tool call, got {sorted(set(tools))}")
        else:
            for tool in expected:
                if tool not in tools:
                    failures.append(f"expected {tool} to be called, got {sorted(set(tools)) or 'none'}")

    for tool in case.get("forbid_tools", []):
        if tool in tools:
            failures.append(f"{tool} should not have been called")

    # Exact argument values, compared case-insensitively as strings.
    for tool, expected_args in case.get("expect_args", {}).items():
        actual = args_for(tool)
        for key, want in expected_args.items():
            got = actual.get(key)
            if got is None:
                failures.append(f"{tool}: missing arg '{key}' (expected {want!r})")
            elif str(got).strip().lower() != str(want).strip().lower():
                failures.append(f"{tool}: arg '{key}' was {got!r}, expected {want!r}")

    for tool, forbidden in case.get("forbid_args", {}).items():
        actual = args_for(tool)
        for key, bad in forbidden.items():
            if str(actual.get(key, "")).strip().lower() == str(bad).strip().lower():
                failures.append(f"{tool}: arg '{key}' must not be {bad!r}")

    # Substring match — the model paraphrases, so exact equality would be brittle.
    for tool, expected_args in case.get("expect_args_contain", {}).items():
        actual = args_for(tool)
        for key, needle in expected_args.items():
            got = actual.get(key)
            if got is None:
                failures.append(f"{tool}: missing arg '{key}' (should mention {needle!r})")
            elif str(needle).lower() not in str(got).lower():
                failures.append(f"{tool}: arg '{key}' was {got!r}, should mention {needle!r}")

    # Present and non-empty. Catches the tool-called-but-args-blank failure.
    for tool, keys in case.get("expect_args_present", {}).items():
        actual = args_for(tool)
        for key in keys:
            value = actual.get(key)
            if value is None or str(value).strip() == "":
                failures.append(f"{tool}: arg '{key}' missing or empty")

    for needle in case.get("expect_reply_contains", []):
        if needle.lower() not in reply.lower():
            failures.append(f"reply should contain {needle!r}")

    if "expect_patient_count_delta" in case:
        want = case["expect_patient_count_delta"]
        got = observed.get("count_delta")
        if got != want:
            failures.append(f"patient count changed by {got}, expected {want}")

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
