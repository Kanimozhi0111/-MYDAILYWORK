import json
import os
import re
from collections import Counter
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def simple_tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.stoi = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = {idx: tok for idx, tok in enumerate(SPECIAL_TOKENS)}

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_idx(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_idx(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.stoi)

    def build(self, captions: List[str]) -> None:
        counter = Counter()
        for cap in captions:
            counter.update(simple_tokenize(cap))
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def numericalize(self, caption: str, max_len: int = 30) -> List[int]:
        tokens = simple_tokenize(caption)[: max_len - 2]
        ids = [self.bos_idx] + [self.stoi.get(t, self.unk_idx) for t in tokens] + [self.eos_idx]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        words = []
        for idx in token_ids:
            tok = self.itos.get(int(idx), "<unk>")
            if tok == "<eos>":
                break
            if tok not in {"<bos>", "<pad>"}:
                words.append(tok)
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "min_freq": self.min_freq}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 1))
        vocab.stoi = {k: int(v) for k, v in data["stoi"].items()}
        vocab.itos = {idx: tok for tok, idx in vocab.stoi.items()}
        return vocab


class CaptionDataset(Dataset):
    def __init__(self, images_dir: str, captions_file: str, vocab: Vocabulary, transform=None, max_len: int = 30):
        self.images_dir = images_dir
        self.df = pd.read_csv(captions_file)
        if "image" not in self.df.columns or "caption" not in self.df.columns:
            raise ValueError("captions_file must contain 'image' and 'caption' columns.")
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption_ids = self.vocab.numericalize(str(row["caption"]), self.max_len)
        caption = torch.tensor(caption_ids, dtype=torch.long)
        return image, caption


def build_vocab_from_csv(captions_file: str, min_freq: int = 2) -> Vocabulary:
    df = pd.read_csv(captions_file)
    if "caption" not in df.columns:
        raise ValueError("captions_file must contain a 'caption' column.")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(df["caption"].astype(str).tolist())
    return vocab


def collate_fn(pad_idx: int):
    def _fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        return images, captions, lengths

    return _fn
