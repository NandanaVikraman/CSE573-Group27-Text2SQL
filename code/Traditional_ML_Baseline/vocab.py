import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TRAIN_PROCESSED_PATH = PROCESSED_DIR / "spider_train_processed.jsonl"
SRC_VOCAB_PATH = PROCESSED_DIR / "src_vocab.json"
TGT_VOCAB_PATH = PROCESSED_DIR / "tgt_vocab.json"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Vocab:
    def __init__(self, tokens=None):
        self.token_to_id = {}
        self.id_to_token = []

        for token in SPECIAL_TOKENS:
            self.add_token(token)

        if tokens:
            for token in tokens:
                self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def encode(self, tokens):
        unk_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids]

    def __len__(self):
        return len(self.id_to_token)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "token_to_id": self.token_to_id,
                    "id_to_token": self.id_to_token,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def load_processed_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_vocabs(data):
    src_counter = Counter()
    tgt_counter = Counter()

    for ex in data:
        src_counter.update(ex["input_tokens"])
        tgt_counter.update(ex["sql_tokens"])

    src_vocab = Vocab(src_counter.keys())
    tgt_vocab = Vocab(tgt_counter.keys())
    return src_vocab, tgt_vocab


if __name__ == "__main__":
    data = load_processed_jsonl(TRAIN_PROCESSED_PATH)
    src_vocab, tgt_vocab = build_vocabs(data)

    src_vocab.save(SRC_VOCAB_PATH)
    tgt_vocab.save(TGT_VOCAB_PATH)

    print(f"Loaded {len(data)} processed examples")
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    print("Encoded sample source:", src_vocab.encode(data[0]["input_tokens"][:10]))
    print("Encoded sample target:", tgt_vocab.encode(data[0]["sql_tokens"]))
