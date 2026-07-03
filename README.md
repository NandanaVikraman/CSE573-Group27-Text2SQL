# CSE 573 — Group 27: Text2SQL with LLM-Modulo

Natural language to SQL translation using an iterative LLM-Modulo verification loop. Evaluated on the Spider and BIRD benchmarks.

> **Note:** This is a personal copy of a team project built for CSE 573 at Arizona State University, hosted here for portfolio purposes. All code was developed collaboratively by the six team members listed below; see the [original repository](https://github.com/AbhinavGor/CSE573-Group27-Text2SQL) for the canonical version and full commit history.

## Team Members

| Member | GitHub |
|---|---|
| Abhinav Gorantla | [@abhinavgorantla](https://github.com/abhinavgorantla) |
| Rohitha Somuri | [@Rohitha21032003](https://github.com/Rohitha21032003) |
| Aryan Brijeshkumar Patel | [@P-Aryan](https://github.com/P-Aryan) |
| Tejas Ajay Parse | [@TejasParse](https://github.com/TejasParse) |
| Nandana Vikraman | [@NandanaVikraman](https://github.com/NandanaVikraman) |
| Kamal Teja Annamdasu | [@kamalteja24](https://github.com/kamalteja24) |

## Repository Contents

Three top-level directories are included:

- **`code/`** — source code for experiments we run (Python backend + React frontend)
  - Traditional_ML_Baseline - Has the code for our tradional ML baseline algorithm
- **`data/`** — Spider and BIRD benchmark datasets
- **`evaluations/`** — pre-run evaluation artifacts organized by experiment type
  - `evaluations/baselines/` — zero-shot baseline runs for `flan-t5`, `flan-t5-base`, `qwen2.5-0.5b`
  - `evaluations/ablations/verifier_ablation/` — modulo runs with each verifier subset: syntax-only, schema-only, judge-only
  - `evaluations/ablations/iter_ablation/` — modulo runs varying `candidates_per_iter` (1, 2, 3)

Each evaluation folder contains `checkpoint.jsonl` (per-sample predictions), `baseline_<timestamp>.json` (summary + full results), and `logs.txt`.

## Overview

The system translates a natural language question and a database schema into a SQL query. It supports two modes:

- **Baseline** — single-pass zero-shot generation
- **LLM-Modulo** — iterative loop that generates SQL candidates and filters them through verifiers until one passes, then returns it

### Verifiers

| Verifier | What it checks |
|---|---|
| `syntax` | Parses SQL with sqlglot; rejects unparseable output |
| `schema` | Checks that all referenced tables and columns exist in the schema |
| `judge` | LLM-as-judge scores the candidate given the question and schema |

### Two-phase pipeline (modulo mode)

1. **Phase 1** — Feed the question and full DB schema to the model; extract only the relevant tables/columns. Result is cached per (model, db, question) key.
2. **Phase 2** — Modulo loop: generate up to `candidates_per_iter` SQL candidates per iteration, run verifiers, return on first all-passing candidate. Falls back to best candidate after `iter_limit` iterations.

## Repository Layout

```
submission-code/
├── code/
│   ├── backend/
│   │   ├── main.py          # CLI: batch evaluation and cache warmup
│   │   ├── api.py           # FastAPI REST server (POST /generate-sql)
│   │   ├── constants.py     # Model aliases and prompt templates
│   │   ├── modules/
│   │   │   ├── llm.py       # Model init, generation, modulo loop
│   │   │   └── verifiers/   # Syntax, schema, and LLM-judge verifiers
│   │   └── utils/
│   │       ├── data_loader.py
│   │       └── metadata_cache.py
│   └── frontend/
│       └── src/
│           └── App.jsx      # React UI (query builder + history)
└── data/
    ├── spider/              # Spider dev + train splits
    └── bird/                # BIRD subset
```

## Setup

**Python >= 3.12** and [uv](https://github.com/astral-sh/uv) are required.

```bash
cd code/backend
uv sync
```

On Linux (HPC / A100), uv automatically pulls the CUDA 12.4 torch wheel. On macOS it falls back to CPU torch.

**Frontend:**

```bash
cd code/frontend
npm install
```

## Running

### REST API

```bash
cd code/backend
TEXT2SQL_MODEL=flan-t5 uvicorn api:app --host 0.0.0.0 --port 8000
```

`TEXT2SQL_MODEL` accepts the same aliases as `--model` (see below). The model loads on the first request.

**Endpoint:** `POST /generate-sql`

```json
{
  "question": "How many customers placed orders this month?",
  "schema_ddl": "CREATE TABLE customers (...); CREATE TABLE orders (...);",
  "use_modulo": true,
  "iter_limit": 3,
  "candidates_per_iter": 3
}
```

### Frontend

```bash
cd code/frontend
npm run dev
```

Set `VITE_API_BASE_URL` to point at the backend (defaults to `http://localhost:8000`).

### CLI Evaluation

```bash
cd code/backend

# Zero-shot baseline on Spider dev
uv run python main.py --mode baseline --dataset-path ../../data/spider/spider_dev_with_schema_new_t1.jsonl --sample-size 100

# LLM-Modulo loop
uv run python main.py --mode modulo --dataset-path ../../data/spider/spider_dev_with_schema_new_t1.jsonl --sample-size 100 --verifiers schema,judge

# Single question (interactive)
uv run python main.py --mode single --question "List all employees hired after 2020" --db-id department_management --model llama3.1

# Pre-warm Phase 1 metadata cache
uv run python main.py --mode cache-warm --dataset-path ../../data/spider/spider_dev_with_schema_new_t1.jsonl
```

### Flags to use when running our code

| Flag | Default | Description |
|---|---|---|
| `--model` | `flan-t5` | Generator model. Aliases: `flan-t5`, `llama3.1`, `qwen`, or any HuggingFace model ID |
| `--judge-model` | (same as generator) | Separate model for LLM-judge verifier |
| `--mode` | `baseline` | `baseline`, `modulo`, `single`, `cache-warm` |
| `--verifiers` | `schema,judge` | Comma-separated subset of `syntax,schema,judge` |
| `--schema-mode` | `query_relevant` | `query_relevant` (Phase 1 filtering) or `full` (entire schema) |
| `--iter-limit` | `3` | Max modulo iterations per sample |
| `--candidates-per-iter` | `3` | SQL candidates generated per iteration |
| `--inference-batch-size` | `16` | GPU batch size; increase for better A100 utilisation |
| `--start-index` / `--end-index` | `0` / `-1` | Slice the dataset for SLURM array sharding |
| `--disable-metadata-cache` | off | Skip Phase 1 cache reads |


## Artifacts

Each run writes to `artifacts/<timestamp>/`:

- `logs.txt` — full run log
- `checkpoint.jsonl` — per-sample results (enables resume on failure)
- `baseline_<timestamp>.json` — final results and summary metrics


## Datasets

| Dataset | Path | Format |
|---|---|---|
| Spider dev | `data/spider/spider_dev_with_schema_new_t1.jsonl` | JSONL |
| Spider train | `data/spider/train_spider.json` | JSON |
| BIRD subset | `data/bird/bird_subset_with_schema_new_t1.jsonl` | JSONL |

We run all model evaluations on Spider train set.