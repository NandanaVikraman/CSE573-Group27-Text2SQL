DEFAULT_MODEL = "juierror/flan-t5-text2sql-with-schema-v2"

# Short aliases accepted by --model; any other value is treated as a full HF model ID.
MODEL_ALIASES: dict[str, str] = {
    "flan-t5": "juierror/flan-t5-text2sql-with-schema-v2",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3.5-9B",
}

DEFAULT_BASELINE_SAMPLE_SIZE = 20
DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_METADATA_CACHE_PATH = "artifacts/cache/phase1_metadata_cache.json"
DEFAULT_ITER_LIMIT = 3
DEFAULT_CANDIDATES_PER_ITER = 3

RELEVANT_METADATA_PROMPT = """Given the database metadata and a question, return the relevant tables and columns that are REQUIRED to answer the given question.
Return JSON only in this format:
{{
  "tables": ["table1", "table2"],
  "columns": {{"table1": ["col1", "col2"], "table2": ["col3"]}}
}}

Metadata:
{metadata_json}

Question:
{question}
"""

LLM_JUDGE_PROMPT = """You are an expert SQL verifier.
Given a natural language question, candidate SQL query, and database metadata, decide if the SQL is likely correct.
Return JSON only in this format:
{{
  "passed": true,
  "confidence": 0.0,
  "rationale": "brief reason"
}}

Question:
{question}

Candidate SQL:
{candidate_sql}

Metadata:
{metadata_json}
"""

SQL_REGENERATION_PROMPT = """You are fixing SQL candidates using verifier feedback.
Given the original task, previous SQL candidates, and verifier failure reasons, produce corrected SQL candidates.
Only return SQL statements separated by new lines, with no explanations.

Original Task:
{base_prompt}

Previous SQL Candidates:
{previous_candidates_json}

Verifier Feedback:
{feedback}
"""
