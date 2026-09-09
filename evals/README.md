# Evals

Behavioural regression tests for the agent loop: tool selection, arguments,
successful execution, and persisted changes to patient and lab records.
These checks complement endpoint tests, which cannot tell whether a model
selected the right tool or supplied the details it claimed to save.

## Running

Offline checks use only the Python standard library and make no model calls:

```bash
python3 -m unittest discover -s evals -p 'test_*.py' -v
python3 evals/test_harness.py
```

For the live suite, install the application's dependencies and set
`OPENAI_API_KEY` in your environment. Then run:

```bash
python3 evals/run_evals.py
python3 evals/run_evals.py --only store_lab_values
python3 evals/run_evals.py --repeat 3 --timeout 300
```

Every live attempt uses the application's model configuration and incurs API
usage. Offline tests do not measure the live model's pass rate. Repeats sample
variation on these specific prompts; they do not estimate a population-wide
user failure rate. Each attempt has a time limit (300 seconds by default).

Exit codes: **0** all attempts passed, **1** at least one behavioural failure,
model-call error or timeout, **2** invalid arguments, missing key, or worker
setup failure. `--repeat` and `--timeout` must be positive integers.

## Isolation and evidence

Every case and every repeat runs in a fresh subprocess with its own temporary
copy of `chroma_seed/` (or an empty store if the seed is absent). Two synthetic
fixture patients are added before the turn. The worker gets both
`MEDAGENT_CHROMA_PATH` and `MEDAGENT_LAB_STORE` before importing the application;
both paths are checked to resolve inside that attempt's temporary directory.
Demo mode is disabled in the worker to exercise all tools.

The worker records each executor call separately, including its result, and
reads patient records from ChromaDB and labs from the JSON file before and after
the turn. It does not treat a claimed success or an in-memory-only update as
proof of persistence. Temporary stores are removed after the worker exits,
including failed and timed-out attempts. No application server is started.

## What the 14 cases cover

- Registration saves the supplied name, age, gender, both symptoms, medication
  dose and allergy, while preserving existing patients.
- Updates affect the intended patient rather than adding a duplicate or
  changing another record.
- Search, listing, analytics and exact-ID retrieval use the appropriate tools
  without changing stored data.
- Lab storage uses the correct action, patient and numeric values, appends one
  complete persisted entry, and preserves existing lab histories.
- Greetings, capability questions and general knowledge avoid tool calls.
- An unconfirmed delete-all preserves complete patient and lab snapshots and
  cannot fall back to single/bulk deletion or fabricated confirmation.
- The crisis case retains the existing response substring regression check.
  It is not a clinical-quality or comprehensive safety evaluation.

Every case is one cold conversation turn. Multi-turn conversations, UI behavior,
retrieval relevance, answer correctness and clinical effectiveness are not fully
covered by this suite. Live-model results must be reported separately from
checker and runner test results.

## Adding a case

Append to `CASES` in `cases.py`. Expected argument constraints for a tool must
all match **one successful call**. Arguments from separate calls are never
combined; every call is checked for forbidden actions.

| Key | Meaning |
|---|---|
| `message`, `patient_id` | User turn and session patient ID; empty ID means none |
| `expect_tools` | Tools that must succeed; `[]` forbids all tool calls |
| `forbid_tools` | Tool names forbidden in every call |
| `expect_args` | Exact values on the same successful call |
| `expect_args_contain` | Required substring, or list of required substrings |
| `expect_args_present` | Nonempty arguments on that same call |
| `forbid_args` | Value, or list of alternative values, forbidden in every call |
| `expect_patient_count_delta` | Required change in persisted patient count |
| `expect_patient_fields` | Patient ID to required persisted field values |
| `expect_new_patient_fields` | Exactly one new record with these field values |
| `expect_unchanged_other_patients` | IDs allowed to change; all other existing patients must be preserved |
| `expect_lab_values` | Patient ID to numeric values in exactly one newly appended lab entry; existing and other patients' histories must be preserved |
| `expect_store_unchanged` | Entire persisted patient and lab snapshots must match |
| `expect_reply_contains` | Required reply substring; use sparingly |
| `why` | Explanation printed on failure |

Exact strings compare without case sensitivity; numbers compare numerically,
without treating booleans as numbers. Persisted patient field expectations may
also use lists of required substrings for free text. Dose substring checks
accept `50mg` and `50 mg` but reject `150mg`.
