# train.py
import argparse
import random
import time
from pathlib import Path

import dynet as dy

dyparams = dy.DynetParams()
dyparams.set_mem(4096)
dyparams.set_requested_gpus(1)
dyparams.init()

from dataset import Text2SQLDataset, Vocab, decode_extended
from model_seq2seq import Seq2SQLModel

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PROCESSED_PATH = BASE_DIR / "processed" / "spider_train_processed.jsonl"
DEV_PROCESSED_PATH = BASE_DIR / "processed" / "spider_dev_processed.jsonl"
SRC_VOCAB_PATH = BASE_DIR / "processed" / "src_vocab.json"
TGT_VOCAB_PATH = BASE_DIR / "processed" / "tgt_vocab.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "model_best.dy"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_latest.dy"
FINAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_final.dy"


def normalize_tokens_to_sql(tokens):
    import re
    sql_str = " ".join(tokens).lower()
    sql_str = re.sub(r"\s+", " ", sql_str).strip()
    return sql_str


def evaluate_exact_match(model, data, tgt_vocab, bos_id, eos_id, max_len=100, beam_size=1):
    results = []
    num_token_exact = 0
    num_normalized_exact = 0

    for sample in data:
        if beam_size > 1:
            pred_ids = model.decode_beam(
                sample["input_ids"],
                bos_id,
                eos_id,
                src_ext_ids=sample["src_ext_ids"],
                num_oov=len(sample["oov_tokens"]),
                beam_size=beam_size,
                max_len=max_len,
            )
        else:
            pred_ids = model.decode_greedy(
                sample["input_ids"],
                bos_id,
                eos_id,
                src_ext_ids=sample["src_ext_ids"],
                num_oov=len(sample["oov_tokens"]),
                max_len=max_len,
            )

        pred_tokens = decode_extended(pred_ids, tgt_vocab, sample["oov_tokens"])
        gold_tokens = sample["sql_tokens"]
        gold_clean = [t for t in gold_tokens if t not in ("<bos>", "<eos>")]

        token_exact = pred_tokens == gold_clean
        normalized_exact = normalize_tokens_to_sql(pred_tokens) == normalize_tokens_to_sql(gold_clean)

        if token_exact:
            num_token_exact += 1
        if normalized_exact:
            num_normalized_exact += 1

        results.append(
            {
                "question": sample["question"],
                "gold_sql": " ".join(gold_clean),
                "pred_sql": " ".join(pred_tokens),
                "token_exact": token_exact,
                "normalized_exact": normalized_exact,
            }
        )

    n = len(data) if data else 0
    token_em = num_token_exact / n if n else 0.0
    normalized_em = num_normalized_exact / n if n else 0.0
    return token_em, normalized_em, results


def main():
    parser = argparse.ArgumentParser(description="Train Text-to-SQL model")
    parser.add_argument("--overfit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_dev", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--dev_data", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--train_dev_subset", type=int, default=100)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=250)
    args = parser.parse_args()

    random.seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_data) if args.train_data else TRAIN_PROCESSED_PATH
    dev_path = Path(args.dev_data) if args.dev_data else DEV_PROCESSED_PATH

    print("Loading vocabularies...")
    src_vocab = Vocab.load(SRC_VOCAB_PATH)
    tgt_vocab = Vocab.load(TGT_VOCAB_PATH)
    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")

    bos_id = tgt_vocab.token_to_id["<bos>"]
    eos_id = tgt_vocab.token_to_id["<eos>"]

    print("Loading datasets...")
    print(f"  Train data: {train_path}")
    print(f"  Dev data: {dev_path}")
    train_dataset = Text2SQLDataset(train_path, src_vocab, tgt_vocab)
    dev_dataset = Text2SQLDataset(dev_path, src_vocab, tgt_vocab)
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Dev examples: {len(dev_dataset)}")

    if args.overfit is not None:
        train_data = [train_dataset[i] for i in range(min(args.overfit, len(train_dataset)))]
        dev_data = train_data
        dev_data_subset = train_data
        checkpoint_name = f"model_overfit{len(train_data)}.dy"
        eval_every = 1
        print(f"\n*** OVERFIT MODE: {len(train_data)} examples ***")
    else:
        train_data = (
            [train_dataset[i] for i in range(min(args.limit_train, len(train_dataset)))]
            if args.limit_train is not None
            else [train_dataset[i] for i in range(len(train_dataset))]
        )
        dev_data = (
            [dev_dataset[i] for i in range(min(args.limit_dev, len(dev_dataset)))]
            if args.limit_dev is not None
            else [dev_dataset[i] for i in range(len(dev_dataset))]
        )
        dev_data_subset = (
            dev_data
            if args.train_dev_subset == -1 or args.train_dev_subset >= len(dev_data)
            else dev_data[: args.train_dev_subset]
        )
        checkpoint_name = "model_best.dy"
        eval_every = max(1, args.eval_every)

        print("\n*** TRAIN/DEV MODE ***")
        print(f"  Training on: {len(train_data)} examples")
        print(f"  Full dev set: {len(dev_data)} examples")
        print(f"  Training-time dev subset: {len(dev_data_subset)} examples")
        print(f"  Eval every: {eval_every} epochs (+ epoch 1 and final)")

    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    latest_checkpoint_path = LATEST_CHECKPOINT_PATH
    final_checkpoint_path = FINAL_CHECKPOINT_PATH

    print("\nInitializing model...")
    model = Seq2SQLModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )
    print(f"  embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}")
    print(f"  Beam size for eval: {args.beam_size}")

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if resume_path.exists():
            print(f"Resuming model from: {resume_path}")
            model.model.populate(str(resume_path))
            print("  Resume checkpoint loaded successfully.")
        else:
            print(f"  WARNING: resume checkpoint not found: {resume_path}")
            print("  Starting from scratch.")

    trainer = dy.AdamTrainer(model.model, alpha=args.lr)
    trainer.set_clip_threshold(args.clip)
    print(f"  AdamTrainer: lr={args.lr}, clip={args.clip}")

    best_dev_metric = 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_data)
        epoch_start = time.time()

        logged_loss_sum = 0.0
        logged_loss_count = 0
        num_tokens = 0

        for i, sample in enumerate(train_data, start=1):
            loss = model.compute_loss(
                sample["input_ids"],
                sample["target_ids"],
                src_ext_ids=sample["src_ext_ids"],
                tgt_ext_ids=sample["tgt_ext_ids"],
                num_oov=len(sample["oov_tokens"]),
            )

            if args.log_every > 0 and (i % args.log_every == 0 or i == len(train_data)):
                logged_loss_sum += loss.value()
                logged_loss_count += 1

            num_tokens += len(sample["target_ids"]) - 1
            loss.backward()
            trainer.update()

            if args.log_every > 0 and i % args.log_every == 0:
                print(f"  epoch {epoch}/{args.epochs} | step {i}/{len(train_data)}")

        avg_logged_loss = logged_loss_sum / logged_loss_count if logged_loss_count > 0 else 0.0
        avg_loss_per_token = logged_loss_sum / num_tokens if num_tokens > 0 else 0.0

        is_first_epoch = epoch == 1
        is_eval_epoch = epoch % eval_every == 0
        is_final_epoch = epoch == args.epochs
        should_eval = is_first_epoch or is_eval_epoch or is_final_epoch

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Logged loss: {avg_logged_loss:.4f} (per-token: {avg_loss_per_token:.4f})")
        print(f"  Epoch time: {time.time() - epoch_start:.1f}s")

        if should_eval:
            token_em, normalized_em, dev_results = evaluate_exact_match(
                model,
                dev_data_subset,
                tgt_vocab,
                bos_id,
                eos_id,
                max_len=args.max_len,
                beam_size=args.beam_size,
            )

            print(f"  Token EM:   {token_em:.4f} ({token_em*100:.2f}%) [debug]")
            print(f"  Norm EM:    {normalized_em:.4f} ({normalized_em*100:.2f}%) [used for checkpointing]")
            if args.beam_size > 1:
                print(f"  (using beam search with beam_size={args.beam_size})")

            if dev_results:
                s = dev_results[0]
                print("  Sample prediction:")
                print(f"    Q: {s['question'][:60]}...")
                print(f"    Gold: {s['gold_sql'][:60]}...")
                print(f"    Pred: {s['pred_sql'][:60]}...")
                print(f"    Token match: {s['token_exact']}")
                print(f"    Normalized match: {s['normalized_exact']}")

            model.model.save(str(latest_checkpoint_path))
            print(f"  -> Saved latest checkpoint: {latest_checkpoint_path.name}")

            if normalized_em > best_dev_metric:
                best_dev_metric = normalized_em
                model.model.save(str(checkpoint_path))
                print(f"  -> Saved best checkpoint (norm EM improved to {best_dev_metric:.4f})")
            elif normalized_em == best_dev_metric and epoch == 1:
                model.model.save(str(checkpoint_path))
                print("  -> Saved initial best checkpoint")
        else:
            next_eval_epoch = min(args.epochs, ((epoch // eval_every) + 1) * eval_every)
            print(f"  Dev eval skipped this epoch (next eval at epoch {next_eval_epoch})")

    model.model.save(str(final_checkpoint_path))

    print("\n" + "=" * 70)
    print("Training complete.")
    print(f"  Best normalized EM: {best_dev_metric:.4f} ({best_dev_metric*100:.2f}%)")
    print(f"  Best checkpoint:    {checkpoint_path}")
    print(f"  Latest checkpoint:  {latest_checkpoint_path}")
    print(f"  Final checkpoint:    {final_checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
