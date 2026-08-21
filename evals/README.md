# Evals

Behavioural tests for the agent loop. They answer one question: **did the model
choose the right tool, and did it actually pass the details it claimed to?**

Endpoint tests cannot answer that. Every endpoint here passed while the agent
was quietly writing empty patient records.

## Running

```bash
python3 evals/test_harness.py                        # checker only, no key, free
OPENAI_API_KEY=sk-... python3 evals/run_evals.py     # full suite, live model
```

```bash
OPENAI_API_KEY=sk-... python3 evals/run_evals.py --only crisis_response
OPENAI_API_KEY=sk-... python3 evals/run_evals.py --repeat 3
```

Exit code is 0 when everything passes, 1 on any failure, 2 on a setup problem.

`--repeat` matters: the agent runs at temperature 0.4, so it is sampled rather
than deterministic. A case that fails one run in five is a real weakness, not
noise — it means roughly one user in five hits it.

## Isolation

The runner copies `chroma_seed/` into a temp directory and points
`MEDAGENT_CHROMA_PATH` and `MEDAGENT_LAB_STORE` at it *before* importing `agent`
and `rag`, which read those paths at import time. It asserts the store really is
inside the temp directory before running anything, and deletes it afterwards.
Your real database is never opened.

## What the cases cover

Three failure modes, drawn from bugs that actually shipped:

- **Wrong tool** — searching when it should be writing a record, or listing every
  patient to answer a question the analytics tool exists for.
- **No tool** — answering from memory instead of reading the database. And the
  inverse: reaching for a tool on a greeting or a general knowledge question.
- **Silent success** — calling the right tool with none of the details, then
  describing those details in the reply anyway. `expect_args_present` and
  `expect_args_contain` exist for this; it is the one that reached production.

Plus two safety cases: an unconfirmed "delete all patients" must leave the store
unchanged, and a crisis message must surface the 988 line.

## Adding a case

Append to `CASES` in `cases.py`. Available keys:

| key | meaning |
|---|---|
| `message` | the user turn |
| `patient_id` | session patient id; `""` for none |
| `expect_tools` | tools that must be called; `[]` asserts none were |
| `forbid_tools` | tools that must not be called |
| `expect_args` | exact arg values, compared case-insensitively |
| `expect_args_contain` | arg must contain this substring |
| `expect_args_present` | arg must exist and be non-empty |
| `forbid_args` | arg must not have this value |
| `expect_reply_contains` | substring that must appear in the reply |
| `expect_patient_count_delta` | required change in patient count |
| `why` | printed on failure, so the reason survives you forgetting it |

Assert on behaviour, not wording. The model paraphrases every run; the tool it
picks and the arguments it passes are what should be stable.

`EVAL-0001` is seeded before the run for cases that need an existing patient.
