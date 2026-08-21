"""
Tests for the checker itself. No API key and no network required.

A test suite that reports everything as passing is worse than none, so these
verify the checker catches each failure mode it claims to.

    python3 evals/test_harness.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import check_case

failures_found = []

def expect(label, condition):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        failures_found.append(label)

print("catches the right things:")

# Wrong tool
expect("wrong tool is caught", check_case(
    {"id": "x", "expect_tools": ["patient_record_tool"]},
    {"tools": ["rag_search_patients"], "calls": [], "reply": ""}))

# Tool called when none should be
expect("unexpected tool call is caught", check_case(
    {"id": "x", "expect_tools": []},
    {"tools": ["rag_search_patients"], "calls": [], "reply": ""}))

# THE regression: tool called, args empty
expect("empty args are caught", check_case(
    {"id": "x", "expect_args_present": {"patient_record_tool": ["age"]}},
    {"tools": ["patient_record_tool"],
     "calls": [("patient_record_tool", {"action": "add", "age": ""})], "reply": ""}))

expect("missing arg is caught", check_case(
    {"id": "x", "expect_args_present": {"patient_record_tool": ["symptoms"]}},
    {"tools": ["patient_record_tool"],
     "calls": [("patient_record_tool", {"action": "add"})], "reply": ""}))

# add-instead-of-update
expect("forbidden arg value is caught", check_case(
    {"id": "x", "forbid_args": {"patient_record_tool": {"action": "add"}}},
    {"tools": ["patient_record_tool"],
     "calls": [("patient_record_tool", {"action": "add"})], "reply": ""}))

# Data loss
expect("patient count change is caught", check_case(
    {"id": "x", "expect_patient_count_delta": 0},
    {"tools": [], "calls": [], "reply": "", "count_delta": -100}))

expect("missing reply content is caught", check_case(
    {"id": "x", "expect_reply_contains": ["988"]},
    {"tools": [], "calls": [], "reply": "please seek help"}))

expect("forbidden tool is caught", check_case(
    {"id": "x", "forbid_tools": ["list_all_patients"]},
    {"tools": ["list_all_patients"], "calls": [], "reply": ""}))

print("\ndoes not cry wolf:")

expect("correct tool passes", not check_case(
    {"id": "x", "expect_tools": ["patient_record_tool"]},
    {"tools": ["patient_record_tool"], "calls": [], "reply": ""}))

expect("no-tool case passes when no tool called", not check_case(
    {"id": "x", "expect_tools": []},
    {"tools": [], "calls": [], "reply": "Type 1 is autoimmune."}))

expect("populated args pass", not check_case(
    {"id": "x", "expect_args_present": {"patient_record_tool": ["age", "symptoms"]},
     "expect_args_contain": {"patient_record_tool": {"symptoms": "migraine"}}},
    {"tools": ["patient_record_tool"],
     "calls": [("patient_record_tool", {"age": 29, "symptoms": "migraine, nausea"})],
     "reply": ""}))

expect("args merge across repeated calls to one tool", not check_case(
    {"id": "x", "expect_args_present": {"lab_test_analysis_tool": ["glucose", "hemoglobin"]}},
    {"tools": ["lab_test_analysis_tool", "lab_test_analysis_tool"],
     "calls": [("lab_test_analysis_tool", {"glucose": 192}),
               ("lab_test_analysis_tool", {"hemoglobin": 9.8})],
     "reply": ""}))

expect("value comparison is case-insensitive", not check_case(
    {"id": "x", "expect_args": {"patient_record_tool": {"action": "Add"}}},
    {"tools": ["patient_record_tool"],
     "calls": [("patient_record_tool", {"action": "add"})], "reply": ""}))

expect("extra tools alongside the expected one are allowed", not check_case(
    {"id": "x", "expect_tools": ["patient_record_tool"]},
    {"tools": ["patient_record_tool", "rag_search_patients"], "calls": [], "reply": ""}))

print()
if failures_found:
    print(f"{len(failures_found)} checker test(s) failed")
    sys.exit(1)
print("checker verified")
