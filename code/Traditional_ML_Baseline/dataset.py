"""
Dataset loading for DyNet Text-to-SQL baseline with extended-vocabulary copy.

The copy mechanism uses an extended vocabulary approach (Pointer-Generator style):
- Base vocabulary: fixed target vocabulary from training
- Extended vocabulary: base vocab + OOV source tokens for each example
- This allows copying source tokens that are not in the fixed vocabulary

For each example, we compute:
- src_ext_ids: maps each source position to extended vocab ID
- tgt_ext_ids: maps each target token to extended vocab ID
- oov_tokens: list of OOV tokens in order of first appearance in source
- num_oov: number of OOV tokens for this example

During decoding, output IDs in range [0, vocab_size) are base vocab tokens,
and IDs in range [vocab_size, vocab_size + num_oov) are OOV tokens that
should be looked up from oov_tokens.
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

PROCESSED_PATH = BASE_DIR / "processed" / "spider_train_processed.jsonl"
SRC_VOCAB_PATH = BASE_DIR / "processed" / "src_vocab.json"
TGT_VOCAB_PATH = BASE_DIR / "processed" / "tgt_vocab.json"

class Vocab:
    def __init__(self, token_to_id, id_to_token):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    @classmethod
    def load(cls, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["token_to_id"], data["id_to_token"])

    def encode(self, tokens):
        unk_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids]

    def __len__(self):
        return len(self.id_to_token)


def build_extended_vocab_mapping(input_tokens, sql_tokens, tgt_vocab):
    """
    Build extended vocabulary mapping for copy mechanism.
    
    The extended vocabulary = base target vocab + OOV source tokens.
    OOV tokens get IDs starting at len(tgt_vocab).
    
    Args:
        input_tokens: List[str] source tokens (question + schema)
        sql_tokens: List[str] target SQL tokens
        tgt_vocab: Vocab object for target vocabulary
    
    Returns:
        src_ext_ids: List[int] extended vocab ID for each source position
        tgt_ext_ids: List[int] extended vocab ID for each target token
        oov_tokens: List[str] OOV tokens in order of first source appearance
    """
    vocab_size = len(tgt_vocab)
    unk_id = tgt_vocab.token_to_id["<unk>"]
    
    # Track OOV tokens from source in order of first appearance
    oov_tokens = []
    oov_to_ext_id = {}  # oov_token -> extended_vocab_id
    
    # Build src_ext_ids: map each source position to extended vocab ID
    src_ext_ids = []
    for token in input_tokens:
        if token in tgt_vocab.token_to_id:
            # Token is in base vocab
            src_ext_ids.append(tgt_vocab.token_to_id[token])
        else:
            # Token is OOV - assign extended vocab ID
            if token not in oov_to_ext_id:
                ext_id = vocab_size + len(oov_tokens)
                oov_to_ext_id[token] = ext_id
                oov_tokens.append(token)
            src_ext_ids.append(oov_to_ext_id[token])
    
    # Build tgt_ext_ids: map each target token to extended vocab ID
    # Target tokens can use base vocab or copy from source (via OOV mapping)
    tgt_ext_ids = []
    for token in sql_tokens:
        if token in tgt_vocab.token_to_id:
            # Token is in base vocab
            tgt_ext_ids.append(tgt_vocab.token_to_id[token])
        elif token in oov_to_ext_id:
            # Token is OOV but appears in source - can be copied
            tgt_ext_ids.append(oov_to_ext_id[token])
        else:
            # Token is truly unknown (not in vocab, not in source)
            # This shouldn't happen often for Text-to-SQL if schema is in source
            tgt_ext_ids.append(unk_id)
    
    return src_ext_ids, tgt_ext_ids, oov_tokens


class Text2SQLDataset:
    def __init__(self, processed_path: Path, src_vocab: Vocab, tgt_vocab: Vocab):
        self.examples = []

        with processed_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Source encoding (for encoder embeddings - uses source vocab)
        input_ids = self.src_vocab.encode(ex["input_tokens"])
        
        # Target encoding for embedding lookup (uses base target vocab)
        # This is used for teacher forcing - previous token embedding
        target_ids = self.tgt_vocab.encode(ex["sql_tokens"])
        
        # Extended vocabulary mapping for copy mechanism
        # src_ext_ids: extended vocab ID for each source position (for copy distribution)
        # tgt_ext_ids: extended vocab ID for each target token (for loss computation)
        # oov_tokens: OOV tokens that can be copied from source
        src_ext_ids, tgt_ext_ids, oov_tokens = build_extended_vocab_mapping(
            ex["input_tokens"], ex["sql_tokens"], self.tgt_vocab
        )

        return {
            "id": ex["id"],
            "db_id": ex["db_id"],
            "question": ex["question"],
            "sql": ex["sql"],
            "input_tokens": ex["input_tokens"],
            "sql_tokens": ex["sql_tokens"],
            # For encoder
            "input_ids": input_ids,
            # For decoder teacher forcing (previous token embedding)
            "target_ids": target_ids,
            # For copy mechanism (extended vocabulary)
            "src_ext_ids": src_ext_ids,   # Extended vocab IDs for source positions
            "tgt_ext_ids": tgt_ext_ids,   # Extended vocab IDs for target tokens (loss)
            "oov_tokens": oov_tokens,     # List of OOV tokens for this example
        }


def decode_extended(pred_ids, tgt_vocab, oov_tokens):
    """
    Decode predicted IDs using extended vocabulary.
    
    Args:
        pred_ids: List[int] predicted token IDs (may include extended IDs)
        tgt_vocab: Vocab object for base target vocabulary
        oov_tokens: List[str] OOV tokens for this example
    
    Returns:
        tokens: List[str] decoded tokens
    """
    vocab_size = len(tgt_vocab)
    tokens = []
    for tid in pred_ids:
        if tid < vocab_size:
            # Base vocabulary token
            tokens.append(tgt_vocab.id_to_token[tid])
        else:
            # Extended vocabulary (OOV) token
            oov_idx = tid - vocab_size
            if oov_idx < len(oov_tokens):
                tokens.append(oov_tokens[oov_idx])
            else:
                # Fallback if index out of range (shouldn't happen)
                tokens.append("<unk>")
    return tokens


if __name__ == "__main__":
    src_vocab = Vocab.load(SRC_VOCAB_PATH)
    tgt_vocab = Vocab.load(TGT_VOCAB_PATH)
    dataset = Text2SQLDataset(PROCESSED_PATH, src_vocab, tgt_vocab)

    print(f"Dataset size: {len(dataset)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    sample = dataset[0]
    print("\n=== Sample 0 ===")
    print("ID:", sample["id"])
    print("Question:", sample["question"])
    print("Input tokens (first 15):", sample["input_tokens"][:15])
    print("Input ids (first 15):", sample["input_ids"][:15])
    print("SQL tokens:", sample["sql_tokens"])
    print("Target ids (base vocab):", sample["target_ids"])
    print("Src ext ids (first 15):", sample["src_ext_ids"][:15])
    print("Tgt ext ids:", sample["tgt_ext_ids"])
    print("OOV tokens:", sample["oov_tokens"])
    
    # Test extended decoding
    print("\nDecoded from tgt_ext_ids:")
    decoded = decode_extended(sample["tgt_ext_ids"], tgt_vocab, sample["oov_tokens"])
    print(decoded)
