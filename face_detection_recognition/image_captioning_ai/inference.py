import argparse

import torch
import torchvision.transforms as T
from PIL import Image

from dataset import Vocabulary
from models import ImageCaptioningModel


def build_argparser():
    parser = argparse.ArgumentParser(description="Generate caption for an image.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="resnet50", choices=["resnet50", "vgg16"])
    parser.add_argument("--decoder", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--max_len", type=int, default=30)
    return parser


@torch.no_grad()
def generate_caption(model, image, vocab: Vocabulary, device, max_len: int = 30):
    model.eval()
    image = image.to(device)
    generated = [vocab.bos_idx]

    for _ in range(max_len):
        in_tokens = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        pad_mask = in_tokens.eq(vocab.pad_idx) if model.decoder_name == "transformer" else None
        logits = model(image, in_tokens, pad_mask=pad_mask)
        next_token = int(logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_token)
        if next_token == vocab.eos_idx:
            break

    return vocab.decode(generated)


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(args.vocab_path)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = ImageCaptioningModel(
        vocab_size=ckpt.get("vocab_size", len(vocab)),
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        embed_dim=ckpt.get("embed_dim", 256),
        hidden_dim=ckpt.get("hidden_dim", 512),
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.1),
        transformer_heads=ckpt.get("transformer_heads", 8),
        transformer_layers=ckpt.get("transformer_layers", 2),
        max_len=ckpt.get("max_len", args.max_len),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(args.image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    caption = generate_caption(model, image, vocab, device, max_len=args.max_len)
    print("Generated caption:")
    print(caption)


if __name__ == "__main__":
    main()
