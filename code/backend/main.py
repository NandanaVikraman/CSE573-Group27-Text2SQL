import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from typing import Any

from constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CANDIDATES_PER_ITER,
    DEFAULT_ITER_LIMIT,
    DEFAULT_METADATA_CACHE_PATH,
    DEFAULT_MODEL,
    MODEL_ALIASES,
)
from modules.llm import (
    build_phase1_cache_key,
    init_judge_model,
    init_model,
    build_table_metadata,
    fetch_relevant_metadata,
    fetch_relevant_metadata_batch,
    generate_sql_candidates,
    generate_sql_candidates_batch,
    get_initialized_model_id,
    get_judge_model_id,
    get_phase1_cache_schema_version,
    metadata_to_prompt_tables,
    run_modulo_loop,
    run_modulo_loop_batch,
)
from utils.metadata_cache import (
    MetadataCacheLockError,
    load_metadata_cache,
    metadata_cache_write_lock,
    write_metadata_cache_atomic,
)

MST = timezone(timedelta(hours=-7), name="MST")

# Set up logging to both console and file in a timestamped directory.
def setup_logging(artifact_dir: str) -> tuple[logging.Logger, str]:
    """
    Args:
        artifact_dir: Base artifacts directory
        
    Returns:
        Tuple of (logger, timestamped_artifact_dir)
    """
    timestamp = datetime.now(MST).strftime('%Y%m%d_%H%M%S')
    timestamped_dir = os.path.join(artifact_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    
    log_file = os.path.join(timestamped_dir, "logs.txt")
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, timestamped_dir


def load_spider_samples(
    dataset_path: str,
    max_samples: int,
    db_id: str | None,
    start_index: int = 0,
    end_index: int = -1,
) -> list[dict[str, Any]]:
    if dataset_path.endswith(".jsonl"):
        rows: list[Any] = []
        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            payload = json.load(dataset_file)
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Unsupported dataset payload format.")

    filtered = [row for row in rows if isinstance(row, dict)]
    for row in filtered:
        if "query" not in row and "gold_sql" in row:
            row["query"] = row["gold_sql"]
    if db_id:
        filtered = [row for row in filtered if row.get("db_id") == db_id]

    if start_index > 0 or end_index >= 0:
        actual_end = end_index if end_index >= 0 else len(filtered)
        filtered = filtered[start_index:actual_end]

    if max_samples <= 0:
        return filtered
    return filtered[:max_samples]


def normalize_sql_for_match(sql: str | None) -> str:
    if not sql:
        return ""
    return " ".join(sql.strip().rstrip(";").split()).lower()


def is_exact_match(predicted_sql: str | None, expected_sql: str | None) -> bool:
    if not predicted_sql or not expected_sql:
        return False
    return normalize_sql_for_match(predicted_sql) == normalize_sql_for_match(expected_sql)


def run_single_question(
    question: str,
    db_id: str | None,
    iter_limit: int,
    candidates_per_iter: int,
    enable_modulo: bool,
) -> dict[str, Any]:
    if enable_modulo:
        if not db_id:
            raise ValueError("--db-id is required when LLM-Modulo is enabled")
        raw_metadata = build_table_metadata(db_id=db_id)
        relevant_metadata = fetch_relevant_metadata(raw_metadata, question)
        prompt_tables = metadata_to_prompt_tables(relevant_metadata)
        result = run_modulo_loop(
            question=question,
            tables=prompt_tables,
            metadata=relevant_metadata,
            iter_limit=iter_limit,
            candidates_per_iter=candidates_per_iter,
        )
        return result

    candidates = generate_sql_candidates(
        question=question,
        tables=None,
        num_candidates=1,
    )
    predicted_sql = candidates[0] if candidates else ""
    return {
        "status": "generated",
        "reason": None,
        "question": question,
        "iterations_used": 1,
        "predicted_sql": predicted_sql,
        "verifiers": None,
        "passing_queries": [],
        "iteration_trace": [],
    }


def _extract_valid_metadata_cache_entries(
    payload: dict[str, Any],
    *,
    logger: logging.Logger,
    cache_path: str,
) -> dict[str, dict[str, Any]] | None:
    expected_version = get_phase1_cache_schema_version()
    cache_version = payload.get("cache_schema_version")
    if cache_version != expected_version:
        logger.warning(
            "Phase 1 cache version mismatch for %s (found %s, expected %s); ignoring cache",
            cache_path,
            cache_version,
            expected_version,
        )
        return None

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        logger.warning(
            "Phase 1 cache at %s is missing a valid entries object; ignoring cache",
            cache_path,
        )
        return None

    valid_entries: dict[str, dict[str, Any]] = {}
    invalid_entries = 0
    for key, entry in raw_entries.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            invalid_entries += 1
            continue

        if not all(isinstance(entry.get(field), str) for field in ("db_id", "question", "model_id")):
            invalid_entries += 1
            continue

        relevant_metadata = entry.get("relevant_metadata")
        if not isinstance(relevant_metadata, dict):
            invalid_entries += 1
            continue

        valid_entries[key] = entry

    if invalid_entries:
        logger.warning(
            "Phase 1 cache at %s has %d invalid entries; ignoring those entries",
            cache_path,
            invalid_entries,
        )

    return valid_entries


def _load_phase1_cache_for_modulo(
    *,
    cache_path: str,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]] | None:
    try:
        payload = load_metadata_cache(cache_path)
    except FileNotFoundError:
        logger.warning(
            "Phase 1 cache file not found at %s; continuing uncached",
            cache_path,
        )
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Failed to load Phase 1 cache from %s (%s); continuing uncached",
            cache_path,
            exc,
        )
        return None

    entries = _extract_valid_metadata_cache_entries(
        payload,
        logger=logger,
        cache_path=cache_path,
    )
    if entries is None:
        return None

    logger.info("Loaded Phase 1 cache: %d entries from %s", len(entries), cache_path)
    return entries


def _load_phase1_cache_for_warmup(
    *,
    cache_path: str,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    try:
        payload = load_metadata_cache(cache_path)
    except FileNotFoundError:
        logger.info("Phase 1 cache does not exist yet at %s; starting fresh", cache_path)
        return {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Failed to load Phase 1 cache from %s (%s); starting with an empty cache",
            cache_path,
            exc,
        )
        return {}

    entries = _extract_valid_metadata_cache_entries(
        payload,
        logger=logger,
        cache_path=cache_path,
    )
    if entries is None:
        logger.warning(
            "Phase 1 cache at %s is incompatible with the current code; starting with an empty cache",
            cache_path,
        )
        return {}

    logger.info("Loaded %d existing Phase 1 cache entries from %s", len(entries), cache_path)
    return entries


def _build_phase1_cache_entry(
    *,
    db_id: str,
    question: str,
    relevant_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "db_id": db_id,
        "question": question,
        "model_id": get_initialized_model_id(),
        "relevant_metadata": relevant_metadata,
    }


def _fetch_phase1_metadata(
    *,
    pending: list[tuple[int, dict[str, Any]]],
    questions: list[str],
    inference_batch_size: int,
    logger: logging.Logger,
    cache_entries: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_metadatas = [build_table_metadata(db_id=s.get("db_id", "")) for _, s in pending]
    cache_keys = [
        build_phase1_cache_key(meta, question) if meta and question.strip() else None
        for meta, question in zip(raw_metadatas, questions)
    ]

    relevant_metadatas: list[dict[str, Any]] = []
    total_hits = 0
    total_misses = 0

    for chunk_start in range(0, len(pending), inference_batch_size):
        chunk_end = chunk_start + inference_batch_size
        chunk_meta = raw_metadatas[chunk_start:chunk_end]
        chunk_q = questions[chunk_start:chunk_end]
        chunk_keys = cache_keys[chunk_start:chunk_end]

        chunk_results: list[dict[str, Any] | None] = [None] * len(chunk_meta)
        miss_meta: list[dict[str, Any]] = []
        miss_q: list[str] = []
        miss_positions: list[int] = []
        chunk_hits = 0
        chunk_misses = 0

        for local_idx, (meta, question, cache_key) in enumerate(zip(chunk_meta, chunk_q, chunk_keys)):
            if not meta or not question.strip():
                chunk_results[local_idx] = meta
                continue

            if cache_entries is not None:
                cached_entry = cache_entries.get(cache_key) if cache_key is not None else None
                if cached_entry is not None:
                    chunk_results[local_idx] = cached_entry["relevant_metadata"]
                    chunk_hits += 1
                    continue
                chunk_misses += 1

            miss_meta.append(meta)
            miss_q.append(question)
            miss_positions.append(local_idx)

        if miss_meta:
            fetched = fetch_relevant_metadata_batch(miss_meta, miss_q)
            for local_idx, metadata in zip(miss_positions, fetched):
                chunk_results[local_idx] = metadata

        relevant_metadatas.extend([
            metadata if metadata is not None else {}
            for metadata in chunk_results
        ])

        if cache_entries is None:
            logger.info(
                f"  metadata: {min(chunk_end, len(pending))}/{len(pending)} done"
            )
        else:
            total_hits += chunk_hits
            total_misses += chunk_misses
            logger.info(
                "  metadata: %d/%d done (cache hits: %d, misses: %d)",
                min(chunk_end, len(pending)),
                len(pending),
                chunk_hits,
                chunk_misses,
            )

    if cache_entries is not None:
        logger.info(
            "Phase 1 cache summary: %d hits, %d misses",
            total_hits,
            total_misses,
        )

    return relevant_metadatas


def run_baseline(
    dataset_path: str,
    sample_size: int,
    artifact_dir: str,
    iter_limit: int,
    candidates_per_iter: int,
    enable_modulo: bool,
    db_id: str | None,
    num_workers: int,
    start_index: int = 0,
    end_index: int = -1,
    inference_batch_size: int = 16,
    metadata_cache_path: str = DEFAULT_METADATA_CACHE_PATH,
    disable_metadata_cache: bool = False,
    schema_mode: str = "query_relevant",
    enabled_verifiers: set[str] | None = None,
) -> dict[str, Any]:
    logger, timestamped_dir = setup_logging(artifact_dir)

    logger.info("Starting baseline evaluation run")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Sample size: {'all' if sample_size <= 0 else sample_size}")
    logger.info(f"Index range: [{start_index}, {'end' if end_index < 0 else end_index})")
    logger.info(f"DB filter: {db_id if db_id else 'none (all dbs)'}")
    logger.info(f"LLM-Modulo enabled: {enable_modulo}")
    logger.info(f"Iter limit: {iter_limit}")
    logger.info(f"Candidates per iteration: {candidates_per_iter}")
    logger.info(f"Inference batch size: {inference_batch_size}")
    if enable_modulo:
        logger.info(f"Phase 1 cache path: {metadata_cache_path}")
        logger.info(f"Phase 1 cache enabled: {not disable_metadata_cache}")
        logger.info(f"Schema mode: {schema_mode}")
        verifiers_str = ",".join(sorted(enabled_verifiers)) if enabled_verifiers else "(none)"
        logger.info(f"Enabled verifiers: {verifiers_str}")
        if enabled_verifiers and "judge" in enabled_verifiers:
            judge_id = get_judge_model_id()
            generator_id = get_initialized_model_id()
            note = " (separate from generator)" if judge_id != generator_id else " (same as generator)"
            logger.info(f"Judge model: {judge_id}{note}")
    logger.info(f"Artifacts will be saved to: {timestamped_dir}")

    checkpoint_path = os.path.join(timestamped_dir, "checkpoint.jsonl")

    samples = load_spider_samples(
        dataset_path=dataset_path,
        max_samples=sample_size,
        db_id=db_id,
        start_index=start_index,
        end_index=end_index,
    )
    logger.info(f"Loaded {len(samples)} samples from dataset")

    results: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    exact_match_count = 0
    total_iterations_used = 0
    schema_pass_count = 0
    judge_pass_count = 0
    all_pass_count = 0

    # Resume from checkpoint
    completed_indices: set[int] = set()
    checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")
    if os.path.getsize(checkpoint_path) > 0 if os.path.exists(checkpoint_path) else False:
        with open(checkpoint_path, "r", encoding="utf-8") as ckpt_read:
            for line in ckpt_read:
                try:
                    row = json.loads(line)
                    results.append(row)
                    completed_indices.add(int(row.get("index", -1)))
                except Exception:
                    pass
        logger.info(f"Resumed from checkpoint: {len(completed_indices)} samples already done")

    pending = [
        (idx, s) for idx, s in enumerate(samples) if idx not in completed_indices
    ]
    logger.info(f"Samples remaining: {len(pending)}")

    questions = [s.get("question", "") for _, s in pending]
    relevant_metadatas: list[dict] = []
    prompt_tables_list: list[dict[str, list[str]] | None]

    if enable_modulo:
        if schema_mode == "full":
            logger.info("Phase 1: skipped (schema_mode=full); feeding entire DB schema to generator")
            relevant_metadatas = [build_table_metadata(db_id=s.get("db_id", "")) for _, s in pending]
        else:
            cache_entries: dict[str, dict[str, Any]] | None = None
            if disable_metadata_cache:
                logger.info("Phase 1 cache reads disabled for this run")
            else:
                cache_entries = _load_phase1_cache_for_modulo(
                    cache_path=metadata_cache_path,
                    logger=logger,
                )

            # Phase 1: batch metadata fetch for all pending samples
            logger.info("Phase 1: fetching relevant metadata (batched)...")
            relevant_metadatas = _fetch_phase1_metadata(
                pending=pending,
                questions=questions,
                inference_batch_size=inference_batch_size,
                logger=logger,
                cache_entries=cache_entries,
            )
        prompt_tables_list = [metadata_to_prompt_tables(m) for m in relevant_metadatas]
    else:
        logger.info("Phase 1: skipped metadata fetch for raw baseline prompting")
        prompt_tables_list = [None] * len(pending)

    # Phase 2: modulo loop or plain generation (batched)
    if enable_modulo:
        logger.info("Phase 2: running batched LLM-Modulo loop...")
        samples_data = [
            {
                "question": questions[k],
                "tables": prompt_tables_list[k],
                "metadata": relevant_metadatas[k],
            }
            for k in range(len(pending))
        ]
        loop_results = run_modulo_loop_batch(
            samples_data=samples_data,
            iter_limit=iter_limit,
            candidates_per_iter=candidates_per_iter,
            inference_batch_size=inference_batch_size,
            progress_callback=logger.info,
            enabled_verifiers=enabled_verifiers,
        )
    else:
        logger.info("Phase 2: generating SQL candidates (batched, no modulo)...")
        gen_items = [
            {
                "question": questions[k],
                "tables": prompt_tables_list[k],
                "num_candidates": 1,
                "feedback": None,
                "previous_candidates": [],
            }
            for k in range(len(pending))
        ]
        loop_results = []
        for chunk_start in range(0, len(pending), inference_batch_size):
            chunk = gen_items[chunk_start : chunk_start + inference_batch_size]
            batch_candidates = generate_sql_candidates_batch(chunk)
            for item_candidates in batch_candidates:
                predicted_sql = item_candidates[0] if item_candidates else ""
                loop_results.append({
                    "status": "generated",
                    "reason": None,
                    "iterations_used": 1,
                    "passing_queries": [],
                    "iteration_trace": [],
                    "predicted_sql": predicted_sql,
                })
            logger.info(
                f"  generation: {min(chunk_start + inference_batch_size, len(pending))}/{len(pending)} done"
            )

    # Assemble final rows, checkpoint, log
    for k, ((orig_index, sample), loop_result) in enumerate(zip(pending, loop_results)):
        question = questions[k]
        expected_sql = sample.get("query")

        if enable_modulo:
            iterations_used = int(loop_result.get("iterations_used", 0))
            if loop_result.get("status") == "success":
                selected = loop_result["passing_queries"][0]
                predicted_sql = selected["sql"]
                verifiers = selected["verifiers"]
                final_reason = None
                logger.info(f"✓ Sample {orig_index + 1}: SUCCESS (iterations: {iterations_used})")
            else:
                predicted_sql = None
                iteration_trace = loop_result.get("iteration_trace", [])
                if iteration_trace:
                    last_candidates = iteration_trace[-1].get("candidates", [])
                    if last_candidates:
                        predicted_sql = last_candidates[0].get("sql")
                verifiers = None
                final_reason = loop_result.get("reason")
                logger.info(
                    f"✗ Sample {orig_index + 1}: FAILURE - {final_reason} (iterations: {iterations_used})"
                )
        else:
            predicted_sql = loop_result.get("predicted_sql", "")
            iterations_used = 1
            verifiers = None
            final_reason = None

        sample_exact_match = is_exact_match(predicted_sql, expected_sql)
        if not enable_modulo:
            if sample_exact_match:
                logger.info(f"✓ Sample {orig_index + 1}: SUCCESS (exact SQL match)")
            else:
                logger.info(f"✗ Sample {orig_index + 1}: FAILURE (exact SQL mismatch)")

        row = {
            "index": orig_index,
            "question": question,
            "expected_sql": expected_sql,
            "status": loop_result.get("status"),
            "reason": final_reason,
            "iterations_used": loop_result.get("iterations_used"),
            "predicted_sql": predicted_sql,
            "is_exact_match": sample_exact_match,
            "verifiers": verifiers,
            "passing_queries": loop_result.get("passing_queries", []),
            "iteration_trace": loop_result.get("iteration_trace", []),
        }
        results.append(row)
        checkpoint_file.write(json.dumps(row) + "\n")
        checkpoint_file.flush()

    checkpoint_file.close()

    results.sort(key=lambda row: int(row.get("index", 0)))

    for row in results:
        iterations_used = int(row.get("iterations_used") or 0)
        total_iterations_used += iterations_used
        exact_match_count += int(row.get("is_exact_match", False))

        if enable_modulo:
            if row.get("status") == "success":
                success_count += 1
                verifiers = row.get("verifiers") or {}
                schema_result = verifiers.get("schema_consistency") or {}
                judge_result = verifiers.get("llm_judge") or {}
                schema_pass_count += int(schema_result.get("passed", False))
                judge_pass_count += int(judge_result.get("passed", False))
                all_pass_count += int(verifiers.get("all_passed", False))
            else:
                failure_count += 1

    # Log final summary
    avg_iterations = (total_iterations_used / len(samples)) if samples else 0.0
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION SUMMARY")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Average iterations used: {avg_iterations:.2f}")
    exact_match_accuracy = (100.0 * exact_match_count / len(samples)) if samples else 0.0
    logger.info(f"Exact-match accuracy: {exact_match_count}/{len(samples)} ({exact_match_accuracy:.2f}%)")
    if enable_modulo:
        logger.info(f"Successes: {success_count} ({100.0 * success_count / len(samples):.1f}%)")
        logger.info(f"Failures: {failure_count} ({100.0 * failure_count / len(samples):.1f}%)")
        logger.info(f"Schema consistency passes: {schema_pass_count}")
        logger.info(f"LLM judge passes: {judge_pass_count}")
        logger.info(f"Both verifiers pass: {all_pass_count}")
    else:
        logger.info(f"Generated SQL outputs: {len(samples)}")
    logger.info("=" * 60)

    summary = {
        "dataset_path": dataset_path,
        "enable_modulo": enable_modulo,
        "db_id": db_id,
        "sample_size": len(samples),
        "iter_limit": iter_limit,
        "candidates_per_iter": candidates_per_iter,
        "success_count": success_count,
        "failure_count": failure_count,
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": exact_match_accuracy,
        "average_iterations_used": avg_iterations,
        "schema_pass_count": schema_pass_count,
        "judge_pass_count": judge_pass_count,
        "both_pass_count": all_pass_count,
    }

    now_mst = datetime.now(MST)
    artifact_payload = {
        "generated_at": now_mst.isoformat(),
        "summary": summary,
        "results": results,
    }

    artifact_name = f"baseline_{now_mst.strftime('%Y%m%d_%H%M%S')}.json"
    artifact_path = os.path.join(timestamped_dir, artifact_name)
    with open(artifact_path, "w", encoding="utf-8") as artifact_file:
        json.dump(artifact_payload, artifact_file, indent=2)
    
    logger.info(f"Artifacts saved to: {timestamped_dir}")
    logger.info(f"Baseline JSON: {artifact_name}")
    logger.info(f"Logs file: logs.txt")

    return {"summary": summary, "artifact_path": artifact_path}


def run_metadata_cache_warmup(
    dataset_path: str,
    sample_size: int,
    artifact_dir: str,
    db_id: str | None,
    start_index: int = 0,
    end_index: int = -1,
    inference_batch_size: int = 16,
    metadata_cache_path: str = DEFAULT_METADATA_CACHE_PATH,
) -> dict[str, Any]:
    logger, timestamped_dir = setup_logging(artifact_dir)

    logger.info("Starting Phase 1 metadata cache warmup")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Sample size: {'all' if sample_size <= 0 else sample_size}")
    logger.info(f"Index range: [{start_index}, {'end' if end_index < 0 else end_index})")
    logger.info(f"DB filter: {db_id if db_id else 'none (all dbs)'}")
    logger.info(f"Inference batch size: {inference_batch_size}")
    logger.info(f"Metadata cache path: {metadata_cache_path}")
    logger.info(f"Artifacts will be saved to: {timestamped_dir}")

    samples = load_spider_samples(
        dataset_path=dataset_path,
        max_samples=sample_size,
        db_id=db_id,
        start_index=start_index,
        end_index=end_index,
    )
    logger.info(f"Loaded {len(samples)} samples from dataset")

    try:
        with metadata_cache_write_lock(metadata_cache_path):
            cache_entries = _load_phase1_cache_for_warmup(
                cache_path=metadata_cache_path,
                logger=logger,
            )

            existing_hits = 0
            skipped_without_prompt = 0
            unique_misses_by_key: dict[str, dict[str, Any]] = {}
            duplicate_miss_refs = 0

            for sample in samples:
                question = str(sample.get("question", ""))
                db_id_value = str(sample.get("db_id", ""))
                raw_metadata = build_table_metadata(db_id=db_id_value)

                if not raw_metadata or not question.strip():
                    skipped_without_prompt += 1
                    continue

                cache_key = build_phase1_cache_key(raw_metadata, question)
                if cache_key in cache_entries:
                    existing_hits += 1
                    continue

                if cache_key in unique_misses_by_key:
                    duplicate_miss_refs += 1
                    continue

                unique_misses_by_key[cache_key] = {
                    "db_id": db_id_value,
                    "question": question,
                    "raw_metadata": raw_metadata,
                }

            unique_miss_items = list(unique_misses_by_key.items())
            logger.info(
                "Warmup summary before generation: %d cached, %d unique misses, %d duplicate miss references, %d skipped without prompt",
                existing_hits,
                len(unique_miss_items),
                duplicate_miss_refs,
                skipped_without_prompt,
            )

            processed_unique_misses = 0
            cache_schema_version = get_phase1_cache_schema_version()
            for chunk_start in range(0, len(unique_miss_items), inference_batch_size):
                chunk = unique_miss_items[chunk_start : chunk_start + inference_batch_size]
                chunk_keys = [cache_key for cache_key, _ in chunk]
                chunk_items = [item for _, item in chunk]
                chunk_meta = [item["raw_metadata"] for item in chunk_items]
                chunk_questions = [item["question"] for item in chunk_items]

                fetched = fetch_relevant_metadata_batch(chunk_meta, chunk_questions)
                for cache_key, item, relevant_metadata in zip(chunk_keys, chunk_items, fetched):
                    cache_entries[cache_key] = _build_phase1_cache_entry(
                        db_id=item["db_id"],
                        question=item["question"],
                        relevant_metadata=relevant_metadata,
                    )

                write_metadata_cache_atomic(
                    metadata_cache_path,
                    cache_schema_version=cache_schema_version,
                    entries=cache_entries,
                )
                processed_unique_misses += len(chunk)
                logger.info(
                    "  cache warm: %d/%d unique misses processed",
                    processed_unique_misses,
                    len(unique_miss_items),
                )

    except MetadataCacheLockError:
        logger.info(
            "Another cache-warm process already holds the writer lock for %s; exiting cleanly",
            metadata_cache_path,
        )
        return {
            "status": "skipped",
            "reason": "metadata_cache_lock_held",
            "cache_path": metadata_cache_path,
        }

    summary = {
        "dataset_path": dataset_path,
        "db_id": db_id,
        "sample_size": len(samples),
        "existing_cache_hits": existing_hits,
        "unique_cache_misses": len(unique_miss_items),
        "duplicate_miss_refs": duplicate_miss_refs,
        "skipped_without_prompt": skipped_without_prompt,
        "final_entry_count": len(cache_entries),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1 CACHE WARMUP SUMMARY")
    logger.info(f"Total samples considered: {len(samples)}")
    logger.info(f"Already cached: {existing_hits}")
    logger.info(f"Unique misses generated: {len(unique_miss_items)}")
    logger.info(f"Duplicate miss references: {duplicate_miss_refs}")
    logger.info(f"Skipped without prompt: {skipped_without_prompt}")
    logger.info(f"Final cache entries: {len(cache_entries)}")
    logger.info("=" * 60)

    return {
        "status": "completed",
        "cache_path": metadata_cache_path,
        "artifact_dir": timestamped_dir,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text2SQL baseline and verifier runner")
    parser.add_argument(
        "--model",
        default="flan-t5",
        help=(
            "Model to use for inference. "
            "Aliases: 'flan-t5' (default), 'llama3.1', 'qwen'. "
            "Or pass any full HuggingFace model ID."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Optional separate model for the LLM-as-judge verifier. "
            "Accepts the same aliases as --model or any full HuggingFace model ID. "
            "If unset (default), the judge reuses the generator model."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["single", "baseline", "modulo", "cache-warm"],
        default="baseline",
        help=(
            "'baseline': raw zero-shot inference from the question only. "
            "'modulo': LLM-Modulo loop with verifier feedback. "
            "'cache-warm': populate the shared Phase 1 metadata cache. "
            "'single': run a single question."
        ),
    )
    parser.add_argument(
        "--question",
        default="How many heads of the departments are older than 56 ?",
        help="Question text for single mode.",
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/spider/train_spider.json",
        help="Path to Spider dataset JSON file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of examples to process (0 means all selected samples).",
    )
    parser.add_argument(
        "--db-id",
        default=None,
        help=(
            "Optional Spider DB id filter (for example: department_management). "
            "Required only for schema-aware modulo runs."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory to write evaluation artifacts.",
    )
    parser.add_argument(
        "--metadata-cache-path",
        default=DEFAULT_METADATA_CACHE_PATH,
        help="Path to the shared Phase 1 metadata cache JSON file.",
    )
    parser.add_argument(
        "--disable-metadata-cache",
        action="store_true",
        help="Disable reads from the shared Phase 1 metadata cache during modulo runs.",
    )
    parser.add_argument(
        "--schema-mode",
        choices=["query_relevant", "full"],
        default="query_relevant",
        help=(
            "Schema fed to the SQL generator in modulo mode. "
            "'query_relevant' (default) uses Phase 1 query-relevant metadata extraction. "
            "'full' skips Phase 1 and feeds the entire DB schema."
        ),
    )
    parser.add_argument(
        "--verifiers",
        default="schema,judge",
        help=(
            "Comma-separated subset of {syntax,schema,judge} for the modulo loop. "
            "Defaults to 'schema,judge' (current behavior). Pass an empty string to "
            "disable all verifiers (every candidate vacuously passes)."
        ),
    )
    parser.add_argument(
        "--iter-limit",
        type=int,
        default=DEFAULT_ITER_LIMIT,
        help="Maximum retries in iterative LLM-Modulo loop.",
    )
    parser.add_argument(
        "--candidates-per-iter",
        type=int,
        default=DEFAULT_CANDIDATES_PER_ITER,
        help="Number of SQL candidates generated per iteration.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Kept for backwards compatibility; ignored (batched GPU inference is used instead).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First sample index to process (inclusive). Used for SLURM array sharding.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Last sample index to process (exclusive). -1 means process to the end.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=16,
        help=(
            "Number of samples to process in each GPU batch. "
            "Increase (e.g. 32, 64) for better A100 utilisation; "
            "decrease if you hit OOM errors."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_id = MODEL_ALIASES.get(args.model, args.model)
    init_model(model_id)

    judge_model_id = (
        MODEL_ALIASES.get(args.judge_model, args.judge_model)
        if args.judge_model
        else None
    )
    init_judge_model(judge_model_id)

    if args.mode == "single":
        logger, _ = setup_logging(args.artifact_dir)
        logger.info(f"Running single-question mode")
        logger.info(f"Question: {args.question}")
        logger.info(f"DB id: {args.db_id if args.db_id else 'none (raw baseline prompt)'}")
        logger.info(f"Iter limit: {args.iter_limit}")
        logger.info(f"Candidates per iteration: {args.candidates_per_iter}")

        result = run_single_question(
            question=args.question,
            db_id=args.db_id,
            iter_limit=args.iter_limit,
            candidates_per_iter=args.candidates_per_iter,
            enable_modulo=False,
        )

        if result.get("status") in {"success", "generated"}:
            logger.info("✓ Query generation succeeded")
        else:
            logger.info(f"✗ Query generation failed: {result.get('reason')}")

        print(json.dumps(result, indent=2))
    elif args.mode == "cache-warm":
        output = run_metadata_cache_warmup(
            dataset_path=args.dataset_path,
            sample_size=args.sample_size,
            artifact_dir=args.artifact_dir,
            db_id=args.db_id,
            start_index=args.start_index,
            end_index=args.end_index,
            inference_batch_size=args.inference_batch_size,
            metadata_cache_path=args.metadata_cache_path,
        )
        print(json.dumps(output, indent=2))
    else:
        enable_modulo = args.mode == "modulo"
        verifiers_raw = (args.verifiers or "").strip()
        enabled_verifiers: set[str] = (
            {v.strip() for v in verifiers_raw.split(",") if v.strip()}
            if verifiers_raw
            else set()
        )
        valid_verifiers = {"syntax", "schema", "judge"}
        unknown = enabled_verifiers - valid_verifiers
        if unknown:
            parser_error = f"--verifiers contains unknown names: {sorted(unknown)}; valid: {sorted(valid_verifiers)}"
            raise SystemExit(parser_error)
        output = run_baseline(
            dataset_path=args.dataset_path,
            sample_size=args.sample_size,
            artifact_dir=args.artifact_dir,
            iter_limit=args.iter_limit,
            candidates_per_iter=args.candidates_per_iter,
            enable_modulo=enable_modulo,
            db_id=args.db_id,
            num_workers=1,
            start_index=args.start_index,
            end_index=args.end_index,
            inference_batch_size=args.inference_batch_size,
            metadata_cache_path=args.metadata_cache_path,
            disable_metadata_cache=args.disable_metadata_cache,
            schema_mode=args.schema_mode,
            enabled_verifiers=enabled_verifiers,
        )
        print(json.dumps(output, indent=2))
