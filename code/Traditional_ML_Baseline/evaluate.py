# evaluate.py
import argparse
import json
import re
import sqlite3
from pathlib import Path

import dynet as dy

dyparams = dy.DynetParams()
dyparams.set_mem(4096)
dyparams.set_requested_gpus(1)
dyparams.init()

from dataset import Text2SQLDataset, Vocab, decode_extended
from model_seq2seq import Seq2SQLModel

BASE_DIR = Path(__file__).resolve().parent
DEV_PROCESSED_PATH = BASE_DIR / "processed" / "spider_dev_processed.jsonl"
SRC_VOCAB_PATH = BASE_DIR / "processed" / "src_vocab.json"
TGT_VOCAB_PATH = BASE_DIR / "processed" / "tgt_vocab.json"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "model_best.dy"
SPIDER_DB_DIR = Path.home() / "SWM" / "spider_data" / "database"


def compute_token_exact_match(pred_tokens, gold_tokens):
    gold_clean = [t for t in gold_tokens if t not in ("<bos>", "<eos>")]
    return pred_tokens == gold_clean


def normalize_tokens_to_sql(tokens):
    sql_str = " ".join(tokens).lower()
    sql_str = re.sub(r"\s+", " ", sql_str).strip()
    return sql_str


def compute_normalized_exact_match(pred_tokens, gold_tokens):
    return normalize_tokens_to_sql(pred_tokens) == normalize_tokens_to_sql(gold_tokens)


def get_db_path(db_id):
    db_path = SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
    return db_path if db_path.exists() else None


def check_syntax_validity_sqlite(sql_str, db_id):
    db_path = get_db_path(db_id)
    if db_path is None:
        return False, f"Database not found: {db_id}"

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            cursor.execute(f"EXPLAIN {sql_str}")
            conn.close()
            return True, None
        except sqlite3.Error:
            pass

        try:
            if sql_str.strip().lower().startswith("select"):
                cursor.execute(f"SELECT * FROM ({sql_str}) LIMIT 0")
            else:
                cursor.execute(sql_str)
            conn.close()
            return True, None
        except sqlite3.Error as e:
            conn.close()
            return False, str(e)
    except sqlite3.Error as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def execute_sql(sql_str, db_id):
    db_path = get_db_path(db_id)
    if db_path is None:
        return False, f"Database not found: {db_id}"

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql_str)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except sqlite3.Error as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def normalize_result_set(results):
    normalized_rows = []
    for row in results:
        normalized_row = []
        for val in row:
            if val is None:
                normalized_row.append(None)
            elif isinstance(val, float):
                normalized_row.append(round(val, 6))
            elif isinstance(val, str):
                normalized_row.append(val.lower().strip())
            else:
                normalized_row.append(val)
        normalized_rows.append(tuple(normalized_row))
    return normalized_rows


def compute_execution_accuracy(pred_sql, gold_sql, db_id):
    gold_success, gold_result = execute_sql(gold_sql, db_id)
    if not gold_success:
        return False, False, False, f"Gold SQL error: {gold_result}"

    pred_success, pred_result = execute_sql(pred_sql, db_id)
    if not pred_success:
        return False, False, True, f"Pred SQL error: {pred_result}"

    return normalize_result_set(pred_result) == normalize_result_set(gold_result), True, True, None


SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "like", "between",
    "is", "null", "true", "false", "as", "on", "join", "inner", "left",
    "right", "outer", "full", "cross", "natural", "using", "group", "by",
    "having", "order", "asc", "desc", "limit", "offset", "union", "all",
    "intersect", "except", "distinct", "count", "sum", "avg", "min", "max",
    "case", "when", "then", "else", "end", "cast", "exists", "any", "some",
    "*", ",", "(", ")", ".", ";", "=", "<", ">", "<=", ">=", "!=", "<>",
    "+", "-", "/", "%", "'", '"', "[", "]", "_",
}


def find_suspicious_tokens(pred_tokens, input_tokens):
    input_set = set(input_tokens)
    suspicious = []
    for token in pred_tokens:
        if token in SQL_KEYWORDS or token in input_set:
            continue
        try:
            float(token)
            continue
        except ValueError:
            suspicious.append(token)
    return suspicious


def save_predictions_for_spider(predictions, gold_sqls, db_ids, pred_path, gold_path=None, db_id_path=None):
    with open(pred_path, "w", encoding="utf-8") as f:
        for sql in predictions:
            f.write(" ".join(sql.split()) + "\n")
    print(f"  Saved {len(predictions)} predictions to: {pred_path}")

    if gold_path:
        with open(gold_path, "w", encoding="utf-8") as f:
            for sql, db_id in zip(gold_sqls, db_ids):
                f.write(f"{' '.join(sql.split())}\t{db_id}\n")
        print(f"  Saved {len(gold_sqls)} gold entries to: {gold_path}")

    if db_id_path:
        with open(db_id_path, "w", encoding="utf-8") as f:
            for db_id in db_ids:
                f.write(db_id + "\n")
        print(f"  Saved {len(db_ids)} db_ids to: {db_id_path}")


def save_results_jsonl(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved detailed results to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL model on dev set")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--dev_data", type=str, default=None)
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument("--save_gold", type=str, default=None)
    parser.add_argument("--save_db_ids", type=str, default=None)
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--skip_exec", action="store_true", help="Skip execution accuracy for speed")
    parser.add_argument("--skip_syntax", action="store_true", help="Skip SQLite syntax checks for speed")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
    dev_data_path = Path(args.dev_data) if args.dev_data else DEV_PROCESSED_PATH

    print("Loading vocabularies...")
    src_vocab = Vocab.load(SRC_VOCAB_PATH)
    tgt_vocab = Vocab.load(TGT_VOCAB_PATH)
    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")

    print("Loading dev dataset...")
    print(f"  Dev data: {dev_data_path}")
    dataset = Text2SQLDataset(dev_data_path, src_vocab, tgt_vocab)
    print(f"  Total dev examples: {len(dataset)}")

    if SPIDER_DB_DIR.exists():
        print(f"  Spider DB dir: {SPIDER_DB_DIR} (found)")
    else:
        print(f"  WARNING: Spider DB dir not found: {SPIDER_DB_DIR}")

    eval_indices = list(range(len(dataset))) if args.limit == -1 else list(range(min(args.limit, len(dataset))))
    print(f"  Evaluating {len(eval_indices)} examples")

    bos_id = tgt_vocab.token_to_id["<bos>"]
    eos_id = tgt_vocab.token_to_id["<eos>"]

    print("\nInitializing model...")
    model = Seq2SQLModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    print(f"  embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}")
    print(f"  Beam size: {args.beam_size}")
    if args.beam_size > 1:
        print(f"  Length penalty: {args.length_penalty}")

    print(f"Loading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"  ERROR: Checkpoint not found at {checkpoint_path}")
        return
    model.model.populate(str(checkpoint_path))
    print("  Checkpoint loaded successfully.")

    print("\nEvaluating...")
    print("  Using beam search decoding" if args.beam_size > 1 else "  Using greedy decoding")
    print("=" * 70)

    results = []
    all_predictions = []
    all_gold_sqls = []
    all_db_ids = []
    num_token_em = 0
    num_normalized_em = 0
    num_syntax_valid = 0
    num_exec_acc = 0
    num_gold_exec_failed = 0

    for idx in eval_indices:
        sample = dataset[idx]

        if args.beam_size > 1:
            pred_ids = model.decode_beam(
                sample["input_ids"],
                bos_id,
                eos_id,
                src_ext_ids=sample["src_ext_ids"],
                num_oov=len(sample["oov_tokens"]),
                beam_size=args.beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty,
            )
        else:
            pred_ids = model.decode_greedy(
                sample["input_ids"],
                bos_id,
                eos_id,
                src_ext_ids=sample["src_ext_ids"],
                num_oov=len(sample["oov_tokens"]),
                max_len=args.max_len,
            )

        pred_tokens = decode_extended(pred_ids, tgt_vocab, sample["oov_tokens"])
        gold_tokens = sample["sql_tokens"]
        gold_clean = [t for t in gold_tokens if t not in ("<bos>", "<eos>")]
        pred_sql = " ".join(pred_tokens)
        gold_sql = sample["sql"]

        all_predictions.append(pred_sql)
        all_gold_sqls.append(gold_sql)
        all_db_ids.append(sample["db_id"])

        if args.export_only:
            continue

        token_em = pred_tokens == gold_clean
        if token_em:
            num_token_em += 1

        normalized_em = compute_normalized_exact_match(pred_tokens, gold_clean)
        if normalized_em:
            num_normalized_em += 1

        syntax_valid, syntax_error = (True, None)
        if not args.skip_syntax:
            syntax_valid, syntax_error = check_syntax_validity_sqlite(pred_sql, sample["db_id"])
            if syntax_valid:
                num_syntax_valid += 1

        exec_acc, pred_exec_ok, gold_exec_ok, exec_error = (False, True, True, None)
        if not args.skip_exec:
            exec_acc, pred_exec_ok, gold_exec_ok, exec_error = compute_execution_accuracy(
                pred_sql, gold_sql, sample["db_id"]
            )
            if exec_acc:
                num_exec_acc += 1
            if not gold_exec_ok:
                num_gold_exec_failed += 1

        suspicious = find_suspicious_tokens(pred_tokens, sample["input_tokens"])

        results.append(
            {
                "idx": idx,
                "db_id": sample["db_id"],
                "question": sample["question"],
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "gold_tokens": gold_clean,
                "pred_tokens": pred_tokens,
                "token_em": token_em,
                "normalized_em": normalized_em,
                "syntax_valid": syntax_valid,
                "syntax_error": syntax_error,
                "exec_acc": exec_acc,
                "pred_exec_ok": pred_exec_ok,
                "gold_exec_ok": gold_exec_ok,
                "exec_error": exec_error,
                "suspicious_tokens": suspicious,
                "oov_tokens": sample["oov_tokens"],
            }
        )

    if args.save_predictions:
        print("\nExporting predictions for official Spider evaluation...")
        save_predictions_for_spider(
            all_predictions,
            all_gold_sqls,
            all_db_ids,
            pred_path=args.save_predictions,
            gold_path=args.save_gold,
            db_id_path=args.save_db_ids,
        )

    if args.save_results and not args.export_only:
        save_results_jsonl(results, args.save_results)

    if args.export_only:
        print("\n" + "=" * 70)
        print("Export complete. Use official Spider evaluator for metrics.")
        print("=" * 70)
        return

    n = len(results)
    token_em_rate = num_token_em / n if n > 0 else 0.0
    normalized_em_rate = num_normalized_em / n if n > 0 else 0.0
    syntax_valid_rate = num_syntax_valid / n if (n > 0 and not args.skip_syntax) else None
    exec_acc_rate = num_exec_acc / n if (n > 0 and not args.skip_exec) else None

    print(f"\nSample Predictions ({min(args.num_samples, len(results))} examples):")
    print("=" * 70)
    for i, result in enumerate(results[: args.num_samples]):
        print(f"\n[{i+1}] Example {result['idx']} | db_id: {result['db_id']}")
        print(f"  Question:      {result['question']}")
        print(f"  Gold SQL:      {result['gold_sql']}")
        print(f"  Pred SQL:      {result['pred_sql']}")
        print(f"  Token EM:      {result['token_em']}  [debug]")
        print(f"  Normalized EM: {result['normalized_em']}")
        if not args.skip_syntax:
            print(f"  Syntax Valid:  {result['syntax_valid']}")
        if not args.skip_exec:
            print(f"  Exec Acc:      {result['exec_acc']}")
        if result["syntax_error"]:
            print(f"  Syntax Error:  {result['syntax_error'][:80]}")
        if result["exec_error"]:
            print(f"  Exec Error:    {result['exec_error'][:80]}")
        if result["suspicious_tokens"]:
            print(f"  Suspicious:    {result['suspicious_tokens']}")
        if result["oov_tokens"]:
            print(f"  OOV tokens:    {result['oov_tokens'][:5]}{'...' if len(result['oov_tokens']) > 5 else ''}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Examples evaluated: {n}")
    print(f"  Normalized EM:      {num_normalized_em:4d}/{n} = {normalized_em_rate:.4f} ({normalized_em_rate*100:5.2f}%)")
    print(f"  Token-level EM:     {num_token_em:4d}/{n} = {token_em_rate:.4f} ({token_em_rate*100:5.2f}%)  [debug]")
    if args.skip_syntax:
        print("  Syntax Validity:    skipped")
    else:
        print(f"  Syntax Validity:    {num_syntax_valid:4d}/{n} = {syntax_valid_rate:.4f} ({syntax_valid_rate*100:5.2f}%)")
    if args.skip_exec:
        print("  Execution Accuracy: skipped")
    else:
        print(f"  Execution Accuracy: {num_exec_acc:4d}/{n} = {exec_acc_rate:.4f} ({exec_acc_rate*100:5.2f}%)")
    print()

    if num_gold_exec_failed > 0:
        print(f"  WARNING: {num_gold_exec_failed} gold SQL queries failed to execute")
    print("=" * 70)

    if not args.skip_syntax:
        syntax_errors = [r for r in results if not r["syntax_valid"]]
        if syntax_errors:
            print(f"\nSyntax Error Breakdown ({len(syntax_errors)} invalid):")
            error_types = {}
            for r in syntax_errors:
                err = r["syntax_error"] or "unknown"
                if "no such table" in err.lower():
                    key = "no such table"
                elif "no such column" in err.lower():
                    key = "no such column"
                elif "syntax error" in err.lower():
                    key = "syntax error"
                elif "database not found" in err.lower():
                    key = "database not found"
                else:
                    key = err[:40]
                error_types[key] = error_types.get(key, 0) + 1
            for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
                print(f"  {count:4d} | {err_type}")

    if not args.skip_exec:
        exec_errors = [r for r in results if r["exec_error"] and r["pred_exec_ok"] is False]
        if exec_errors:
            print(f"\nExecution Error Breakdown ({len(exec_errors)} pred SQL failed):")
            error_types = {}
            for r in exec_errors:
                err = r["exec_error"] or "unknown"
                if "no such table" in err.lower():
                    key = "no such table"
                elif "no such column" in err.lower():
                    key = "no such column"
                elif "syntax error" in err.lower():
                    key = "syntax error"
                elif "database not found" in err.lower():
                    key = "database not found"
                else:
                    key = err[:40]
                error_types[key] = error_types.get(key, 0) + 1
            for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:10]:
                print(f"  {count:4d} | {err_type}")

    print()


if __name__ == "__main__":
    main()
