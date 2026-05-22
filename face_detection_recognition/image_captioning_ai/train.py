import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import CaptionDataset, Vocabulary, build_vocab_from_csv, collate_fn
from models import ImageCaptioningModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser():
    parser = argparse.ArgumentParser(description="Train image captioning model (CNN + LSTM/Transformer).")
    parser.add_argument("--images_dir", type=str, default=TrainConfig.images_dir)
    parser.add_argument("--captions_file", type=str, default=TrainConfig.captions_file)
    parser.add_argument("--encoder", type=str, default=TrainConfig.encoder_name, choices=["resnet50", "vgg16"])
    parser.add_argument("--decoder", type=str, default=TrainConfig.decoder_name, choices=["lstm", "transformer"])
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--embed_dim", type=int, default=TrainConfig.embed_dim)
    parser.add_argument("--hidden_dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--num_layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--max_len", type=int, default=TrainConfig.max_len)
    parser.add_argument("--min_word_freq", type=int, default=TrainConfig.min_word_freq)
    parser.add_argument("--transformer_heads", type=int, default=TrainConfig.transformer_heads)
    parser.add_argument("--transformer_layers", type=int, default=TrainConfig.transformer_layers)
    parser.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--artifacts_dir", type=str, default=TrainConfig.artifacts_dir)
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.artifacts_dir, exist_ok=True)

    vocab: Vocabulary = build_vocab_from_csv(args.captions_file, min_freq=args.min_word_freq)
    vocab_path = os.path.join(args.artifacts_dir, TrainConfig.vocab_name)
    vocab.save(vocab_path)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CaptionDataset(
        images_dir=args.images_dir,
        captions_file=args.captions_file,
        vocab=vocab,
        transform=transform,
        max_len=args.max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn(vocab.pad_idx),
    )

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        max_len=args.max_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, captions, _ in pbar:
            images = images.to(device)
            captions = captions.to(device)

            captions_in = captions[:, :-1]
            captions_out = captions[:, 1:]
            pad_mask = captions_in.eq(vocab.pad_idx) if args.decoder == "transformer" else None

            logits = model(images, captions_in, pad_mask=pad_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch}: average loss = {avg_loss:.4f}")

    ckpt_path = os.path.join(args.artifacts_dir, TrainConfig.checkpoint_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder": args.encoder,
            "decoder": args.decoder,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "max_len": args.max_len,
            "transformer_heads": args.transformer_heads,
            "transformer_layers": args.transformer_layers,
            "vocab_size": len(vocab),
        },
        ckpt_path,
    )
    print(f"Saved model to: {ckpt_path}")
    print(f"Saved vocabulary to: {vocab_path}")


if __name__ == "__main__":
    main()
