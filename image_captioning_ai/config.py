from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    images_dir: str = "data/images"
    captions_file: str = "data/captions.csv"

    # Model
    encoder_name: str = "resnet50"  # choices: resnet50, vgg16
    decoder_name: str = "lstm"  # choices: lstm, transformer
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 1
    dropout: float = 0.2
    max_len: int = 30
    transformer_heads: int = 8
    transformer_layers: int = 2

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 10
    min_word_freq: int = 2
    num_workers: int = 0
    seed: int = 42

    # Output
    artifacts_dir: str = "artifacts"
    checkpoint_name: str = "model.pt"
    vocab_name: str = "vocab.json"
