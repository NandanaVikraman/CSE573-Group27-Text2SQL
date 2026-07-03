"""
Preprocessing script for Spider Text-to-SQL data.

Produces model-ready JSONL files for both train and dev splits.
- Train: from raw Spider train JSON + tables.json
- Dev: from enriched JSONL with gold_sql and schema_new already present

Features:
1. Full schema mode: serialize entire database schema (DEFAULT)
2. Schema pruning mode: lexical overlap heuristic to select relevant tables/columns
3. Value grounding (optional, OFF by default): tag question tokens that match schema

Schema Pruning Strategy (classical lexical overlap):
- Tokenize question into words
- Score each table by: # of question tokens appearing in table name or any column name
- Keep tables with score > 0, OR top-k tables if all score 0
- For each kept table, keep columns that match question tokens + primary keys
- Always keep at least min_tables and min_columns to avoid empty schema

Value Grounding Strategy (optional):
- Mark question tokens that exactly match table/column names
- Add special [SCHEMA] tags around matched tokens in input
- This helps the copy mechanism identify copyable schema terms

================================================================================
WARNING: If you use --schema_mode pruned or --value_grounding, you MUST rebuild
the vocabulary from the newly processed training data before training!

The vocabulary files (src_vocab.json, tgt_vocab.json) must match the token
distribution in the processed data. Using mismatched vocab will cause errors
or poor performance.

Steps after changing preprocessing mode:
1. Run preprocess.py with new flags
2. Run build_vocab.py on the new processed training data
3. Then run train.py
================================================================================

DyNet 2.1.2 compatible preprocessing.
Python 3.8+ compatible.
"""

import json
import re
from pathlib import Path


# === PATHS ===
# === PATHS (portable / relative) ===
BASE_DIR = Path(__file__).resolve().parent

# Default input paths (can be overridden via CLI)
TRAIN_INPUT_PATH = BASE_DIR / "data" / "spider" / "train_spider.json"
TABLES_PATH = BASE_DIR / "data" / "spider" / "tables.json"

DEV_INPUT_PATH = BASE_DIR / "data" / "processed" / "spider_dev_with_schema_new_t1.jsonl"

# Output paths (always inside repo)
PROCESSED_DIR = BASE_DIR / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_DIR / "spider_train_processed.jsonl"
DEV_OUTPUT_PATH = PROCESSED_DIR / "spider_dev_processed.jsonl"

# === SCHEMA PRUNING CONFIG ===
DEFAULT_MIN_TABLES = 1      # Minimum tables to keep even if no matches
DEFAULT_MIN_COLUMNS = 2     # Minimum columns per table to keep
DEFAULT_TOP_K_TABLES = 3    # If no matches, keep top-k tables by size


# === LOADERS ===
def load_data(path: Path):
    """Load data from JSON or JSONL file."""
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported file format: {path}")


def save_jsonl(path: Path, data):
    """Save data as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# === TOKENIZATION ===
def tokenize_text(text: str):
    """Tokenize natural language text (question, schema)."""
    text = text.lower().strip()
    return re.findall(r"\w+|[^\w\s]", text)


def tokenize_sql(sql: str):
    """Tokenize SQL query, handling multi-char operators."""
    sql = sql.lower().strip()
    return re.findall(r"\w+|!=|<=|>=|<>|[^\w\s]", sql)


# === SCHEMA BUILDING ===
def load_tables_map(path: Path):
    """Load tables.json and build schema_new for each database."""
    with path.open("r", encoding="utf-8") as f:
        tables_data = json.load(f)

    tables_map = {}

    for db in tables_data:
        db_id = db["db_id"]

        schema_new = {"tables": {}, "foreign_keys": []}

        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        primary_keys = set(db["primary_keys"])
        foreign_keys = db["foreign_keys"]

        for table_name in table_names:
            schema_new["tables"][table_name] = {
                "columns": {},
                "primary_keys": []
            }

        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:
                continue

            table_name = table_names[table_idx]
            col_type = column_types[col_idx]

            schema_new["tables"][table_name]["columns"][col_name] = col_type

            if col_idx in primary_keys:
                schema_new["tables"][table_name]["primary_keys"].append(col_name)

        for src_idx, tgt_idx in foreign_keys:
            src_table_idx, src_col = column_names[src_idx]
            tgt_table_idx, tgt_col = column_names[tgt_idx]

            schema_new["foreign_keys"].append({
                "source_table": table_names[src_table_idx],
                "source_column": src_col,
                "target_table": table_names[tgt_table_idx],
                "target_column": tgt_col,
            })

        tables_map[db_id] = schema_new

    return tables_map


# === NORMALIZATION ===
def normalize_row(row: dict, tables_map: dict = None):
    """
    Normalize a row to have consistent fields: db_id, question, gold_sql, schema_new.
    
    For enriched dev data: gold_sql and schema_new already exist.
    For raw train data: derive gold_sql from 'query', schema_new from tables_map.
    """
    if "gold_sql" in row and "schema_new" in row:
        return {
            "id": row.get("id", -1),
            "db_id": row["db_id"],
            "question": row["question"],
            "gold_sql": row["gold_sql"],
            "schema_new": row["schema_new"],
        }

    if tables_map is None:
        raise ValueError("tables_map required for raw Spider data without schema_new")

    return {
        "id": row.get("id", -1),
        "db_id": row["db_id"],
        "question": row["question"],
        "gold_sql": row["query"],
        "schema_new": tables_map[row["db_id"]],
    }


# === SCHEMA SERIALIZATION ===
def serialize_schema(schema_new: dict) -> str:
    """Convert schema_new dict to flat text representation."""
    parts = []

    tables = schema_new.get("tables", {})
    for table_name, table_info in tables.items():
        columns = list(table_info.get("columns", {}).keys())
        column_str = " , ".join(columns)
        parts.append(f"{table_name} : {column_str}")

    return " ; ".join(parts)


# === SCHEMA PRUNING ===
def compute_lexical_overlap(question_tokens: list, name: str) -> int:
    """
    Compute lexical overlap score between question tokens and a name.
    
    Splits name by common delimiters (underscore, camelCase) and counts
    how many question tokens appear in the name parts.
    
    Args:
        question_tokens: List[str] lowercased question tokens
        name: str table or column name
        
    Returns:
        int overlap score (number of matching tokens)
    """
    # Split name into parts by underscore, then by camelCase boundaries
    name_lower = name.lower()
    # Split by underscore
    parts = name_lower.split('_')
    # Further split camelCase (e.g., "firstName" -> ["first", "name"])
    expanded_parts = []
    for part in parts:
        # Insert space before capital letters for camelCase
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', part).split()
        expanded_parts.extend(camel_split)
    
    # Create set of name parts
    name_tokens = set(expanded_parts)
    
    # Also add the full name (for single-word matches)
    name_tokens.add(name_lower)
    
    # Count overlapping tokens
    question_set = set(question_tokens)
    overlap = len(question_set & name_tokens)
    
    return overlap


def prune_schema(schema_new: dict, question: str, 
                 min_tables: int = DEFAULT_MIN_TABLES,
                 min_columns: int = DEFAULT_MIN_COLUMNS,
                 top_k_tables: int = DEFAULT_TOP_K_TABLES) -> dict:
    """
    Prune schema to keep only tables/columns relevant to the question.
    
    Uses lexical overlap between question tokens and table/column names
    to select relevant schema elements. This is a classical heuristic
    approach that doesn't require embeddings or pretrained models.
    
    Args:
        schema_new: dict with 'tables' and 'foreign_keys'
        question: str natural language question
        min_tables: int minimum tables to keep
        min_columns: int minimum columns per table
        top_k_tables: int fallback number of tables if no matches
        
    Returns:
        pruned_schema: dict with same structure, but filtered
    """
    # Tokenize question
    question_tokens = tokenize_text(question)
    question_set = set(question_tokens)
    
    tables = schema_new.get("tables", {})
    if not tables:
        return schema_new
    
    # Score each table by lexical overlap
    table_scores = {}
    table_column_scores = {}
    
    for table_name, table_info in tables.items():
        # Table name overlap
        table_overlap = compute_lexical_overlap(question_tokens, table_name)
        
        # Column overlaps
        columns = table_info.get("columns", {})
        column_scores = {}
        for col_name in columns.keys():
            col_overlap = compute_lexical_overlap(question_tokens, col_name)
            column_scores[col_name] = col_overlap
        
        # Total table score = table name overlap + sum of column overlaps
        total_col_overlap = sum(column_scores.values())
        table_scores[table_name] = table_overlap + total_col_overlap
        table_column_scores[table_name] = column_scores
    
    # Select tables to keep
    # Keep tables with score > 0
    matched_tables = [t for t, score in table_scores.items() if score > 0]
    
    if len(matched_tables) < min_tables:
        # Fallback: add top-scoring tables until we have min_tables
        sorted_tables = sorted(table_scores.keys(), 
                               key=lambda t: (table_scores[t], len(tables[t].get("columns", {}))),
                               reverse=True)
        for t in sorted_tables:
            if t not in matched_tables:
                matched_tables.append(t)
                if len(matched_tables) >= min_tables:
                    break
    
    # Limit to top_k_tables if we have more
    if len(matched_tables) > top_k_tables:
        matched_tables = sorted(matched_tables, 
                                key=lambda t: table_scores[t], 
                                reverse=True)[:top_k_tables]
    
    # Build pruned schema
    pruned_tables = {}
    for table_name in matched_tables:
        table_info = tables[table_name]
        columns = table_info.get("columns", {})
        primary_keys = table_info.get("primary_keys", [])
        column_scores = table_column_scores[table_name]
        
        # Keep columns with overlap > 0, plus primary keys
        matched_columns = {col for col, score in column_scores.items() if score > 0}
        matched_columns.update(primary_keys)
        
        # Ensure minimum columns
        if len(matched_columns) < min_columns:
            sorted_cols = sorted(columns.keys(), 
                                key=lambda c: column_scores.get(c, 0), 
                                reverse=True)
            for col in sorted_cols:
                matched_columns.add(col)
                if len(matched_columns) >= min_columns:
                    break
        
        # If still not enough columns, keep all
        if len(matched_columns) < min_columns:
            matched_columns = set(columns.keys())
        
        # Build pruned table
        pruned_columns = {col: col_type for col, col_type in columns.items() 
                          if col in matched_columns}
        pruned_tables[table_name] = {
            "columns": pruned_columns,
            "primary_keys": [pk for pk in primary_keys if pk in matched_columns]
        }
    
    # Keep foreign keys that reference kept tables
    pruned_fks = []
    kept_tables_set = set(pruned_tables.keys())
    for fk in schema_new.get("foreign_keys", []):
        if (fk["source_table"] in kept_tables_set and 
            fk["target_table"] in kept_tables_set):
            pruned_fks.append(fk)
    
    return {
        "tables": pruned_tables,
        "foreign_keys": pruned_fks
    }


# === VALUE GROUNDING ===
def ground_schema_in_question(question_tokens: list, schema_new: dict) -> list:
    """
    Mark question tokens that match schema elements for better copy signal.
    
    Wraps matched tokens with [SCHEMA] tags to help the model identify
    copyable schema terms. This is a simple lexical grounding approach
    that doesn't require embeddings.
    
    Args:
        question_tokens: List[str] tokenized question
        schema_new: dict with 'tables' containing column names
        
    Returns:
        grounded_tokens: List[str] tokens with [SCHEMA] markers
    """
    # Collect all schema terms (table names and column names)
    schema_terms = set()
    tables = schema_new.get("tables", {})
    for table_name, table_info in tables.items():
        # Add table name and its parts
        schema_terms.add(table_name.lower())
        for part in table_name.lower().split('_'):
            if len(part) > 1:  # Skip single chars
                schema_terms.add(part)
        
        # Add column names and their parts
        for col_name in table_info.get("columns", {}).keys():
            schema_terms.add(col_name.lower())
            for part in col_name.lower().split('_'):
                if len(part) > 1:
                    schema_terms.add(part)
    
    # Ground question tokens
    grounded_tokens = []
    i = 0
    while i < len(question_tokens):
        token = question_tokens[i]
        token_lower = token.lower() if isinstance(token, str) else token
        
        # Check if this token matches a schema term
        if token_lower in schema_terms:
            # Check for multi-token matches (e.g., "first name" -> "first_name")
            # Look ahead for compound matches
            found_compound = False
            for lookahead in range(min(3, len(question_tokens) - i), 0, -1):
                if lookahead == 1:
                    continue
                compound = "_".join(question_tokens[i:i+lookahead]).lower()
                if compound in schema_terms:
                    # Found compound match - wrap entire compound
                    grounded_tokens.append("[SCHEMA]")
                    grounded_tokens.extend(question_tokens[i:i+lookahead])
                    grounded_tokens.append("[/SCHEMA]")
                    i += lookahead
                    found_compound = True
                    break
            
            if not found_compound:
                # Single token match
                grounded_tokens.append("[SCHEMA]")
                grounded_tokens.append(token)
                grounded_tokens.append("[/SCHEMA]")
                i += 1
        else:
            grounded_tokens.append(token)
            i += 1
    
    return grounded_tokens


# === FINAL EXAMPLE BUILDER ===
def build_example(row: dict, schema_mode: str = "full", value_grounding: bool = False):
    """
    Build final model-ready example with all required fields.
    
    Output fields:
    - id: example identifier
    - db_id: database identifier
    - question: natural language question
    - sql: gold SQL query
    - schema_text: serialized schema string
    - input_tokens: tokenized [question [sep] schema]
    - sql_tokens: tokenized [<bos> sql <eos>]
    
    Args:
        row: dict with id, db_id, question, gold_sql, schema_new
        schema_mode: "full" or "pruned" - whether to use full or pruned schema
        value_grounding: bool - whether to add [SCHEMA] markers to question tokens
    """
    question = row["question"]
    gold_sql = row["gold_sql"]
    schema_new = row["schema_new"]
    
    # Optionally prune schema based on lexical overlap with question
    if schema_mode == "pruned":
        schema_for_input = prune_schema(schema_new, question)
    else:
        schema_for_input = schema_new
    
    schema_text = serialize_schema(schema_for_input)
    
    # Tokenize question
    question_tokens = tokenize_text(question)
    
    # Optionally apply value grounding (mark schema matches in question)
    if value_grounding:
        question_tokens = ground_schema_in_question(question_tokens, schema_for_input)
    
    schema_tokens = tokenize_text(schema_text)
    input_tokens = question_tokens + ["[sep]"] + schema_tokens
    sql_tokens = ["<bos>"] + tokenize_sql(gold_sql) + ["<eos>"]

    return {
        "id": row["id"],
        "db_id": row["db_id"],
        "question": question,
        "sql": gold_sql,
        "schema_text": schema_text,
        "input_tokens": input_tokens,
        "sql_tokens": sql_tokens,
    }


# === SPLIT PROCESSING ===
def process_split(input_path: Path, output_path: Path, tables_path: Path = None,
                  schema_mode: str = "full", value_grounding: bool = False):
    """
    Process a data split (train or dev) and save as model-ready JSONL.
    
    Args:
        input_path: Path to input data (JSON or JSONL)
        output_path: Path for output JSONL
        tables_path: Path to tables.json (required for raw Spider data without schema_new)
        schema_mode: "full" or "pruned" - whether to use full or pruned schema
        value_grounding: bool - whether to add [SCHEMA] markers to question tokens
    """
    print(f"Processing: {input_path}")
    print(f"  Schema mode: {schema_mode}")
    print(f"  Value grounding: {value_grounding}")
    
    rows = load_data(input_path)
    
    tables_map = None
    if tables_path is not None:
        tables_map = load_tables_map(tables_path)

    normalized_rows = []
    for i, row in enumerate(rows):
        normalized = normalize_row(row, tables_map)
        if normalized["id"] == -1:
            normalized["id"] = i
        normalized_rows.append(normalized)

    processed = [build_example(row, schema_mode=schema_mode, value_grounding=value_grounding) 
                 for row in normalized_rows]

    save_jsonl(output_path, processed)

    print(f"  Loaded {len(rows)} raw rows")
    print(f"  Saved {len(processed)} processed rows to: {output_path}")
    if processed:
        print(f"  First example question: {processed[0]['question'][:60]}...")
        print(f"  First example input tokens (first 20): {processed[0]['input_tokens'][:20]}...")
        print(f"  First example SQL tokens: {processed[0]['sql_tokens'][:10]}...")


# === MAIN ===
def main():
    """
    Process both train and dev splits.
    
    Usage:
        python preprocess.py                          # Full schema, no grounding
        python preprocess.py --schema_mode pruned     # Pruned schema
        python preprocess.py --value_grounding        # Add [SCHEMA] markers
        python preprocess.py --schema_mode pruned --value_grounding  # Both
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Spider data for Text-to-SQL")
    parser.add_argument("--schema_mode", type=str, default="full", choices=["full", "pruned"],
                        help="Schema mode: 'full' (default) or 'pruned' (lexical overlap)")
    parser.add_argument("--value_grounding", action="store_true",
                        help="Add [SCHEMA] markers to question tokens matching schema terms")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Spider Text-to-SQL Preprocessing")
    print("=" * 70)
    print(f"  Schema mode: {args.schema_mode}")
    print(f"  Value grounding: {args.value_grounding}")
    
    # Determine output paths based on mode
    # Python 3.8 compatible path construction (Path.with_stem requires 3.9+)
    if args.schema_mode == "pruned" or args.value_grounding:
        suffix = ""
        if args.schema_mode == "pruned":
            suffix += "_pruned"
        if args.value_grounding:
            suffix += "_grounded"
        # Construct new path: parent / (stem + suffix + extension)
        train_output = TRAIN_OUTPUT_PATH.parent / (TRAIN_OUTPUT_PATH.stem + suffix + TRAIN_OUTPUT_PATH.suffix)
        dev_output = DEV_OUTPUT_PATH.parent / (DEV_OUTPUT_PATH.stem + suffix + DEV_OUTPUT_PATH.suffix)
        
        # Print vocab rebuild warning
        print("\n" + "!" * 70)
        print("WARNING: You are using non-default preprocessing options.")
        print("         You MUST rebuild vocabulary from the new processed train data!")
        print("         Run build_vocab.py on the new train JSONL before training.")
        print("!" * 70)
    else:
        train_output = TRAIN_OUTPUT_PATH
        dev_output = DEV_OUTPUT_PATH
    
    # Process train split (raw Spider JSON + tables.json)
    print("\n[1/2] Processing TRAIN split...")
    process_split(
        input_path=TRAIN_INPUT_PATH,
        output_path=train_output,
        tables_path=TABLES_PATH,
        schema_mode=args.schema_mode,
        value_grounding=args.value_grounding,
    )
    
    # Process dev split (enriched JSONL with schema_new already present)
    print("\n[2/2] Processing DEV split...")
    process_split(
        input_path=DEV_INPUT_PATH,
        output_path=dev_output,
        tables_path=None,
        schema_mode=args.schema_mode,
        value_grounding=args.value_grounding,
    )
    
    print("\n" + "=" * 70)
    print("Preprocessing complete.")
    print(f"  Train output: {train_output}")
    print(f"  Dev output: {dev_output}")
    
    # Remind about vocab if non-default mode
    if args.schema_mode == "pruned" or args.value_grounding:
        print("\n  REMINDER: Rebuild vocab from the new train data before training!")
    
    print("=" * 70)


if __name__ == "__main__":
    main()