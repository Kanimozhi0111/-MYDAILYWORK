import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class CNNEncoder(nn.Module):
    def __init__(self, encoder_name: str = "resnet50", embed_dim: int = 256):
        super().__init__()
        self.encoder_name = encoder_name.lower()
        if self.encoder_name == "resnet50":
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            modules = list(backbone.children())[:-1]
            self.cnn = nn.Sequential(*modules)
        elif self.encoder_name == "vgg16":
            backbone = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
            self.cnn = backbone.features
            in_features = 512 * 7 * 7
        else:
            raise ValueError("encoder_name must be one of: resnet50, vgg16")

        for p in self.cnn.parameters():
            p.requires_grad = False

        self.project = nn.Linear(in_features, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(images)
        feats = feats.flatten(start_dim=1)
        return self.project(feats)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_feats: torch.Tensor, captions_in: torch.Tensor) -> torch.Tensor:
        x = self.embed(captions_in)
        h0 = self.init_h(image_feats).unsqueeze(0)
        c0 = self.init_c(image_feats).unsqueeze(0)
        out, _ = self.lstm(x, (h0, c0))
        logits = self.fc(out)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, image_feats: torch.Tensor, captions_in: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        tgt = self.embed(captions_in)
        tgt = self.pos(tgt)
        memory = image_feats.unsqueeze(1)
        seq_len = captions_in.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=captions_in.device), diagonal=1).bool()
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
        )
        logits = self.fc(out)
        return logits


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_name: str = "resnet50",
        decoder_name: str = "lstm",
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        max_len: int = 64,
    ):
        super().__init__()
        self.decoder_name = decoder_name.lower()
        self.encoder = CNNEncoder(encoder_name=encoder_name, embed_dim=embed_dim)
        if self.decoder_name == "lstm":
            self.decoder = LSTMDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif self.decoder_name == "transformer":
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
                max_len=max_len,
            )
        else:
            raise ValueError("decoder_name must be one of: lstm, transformer")

    def forward(self, images: torch.Tensor, captions_in: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        image_feats = self.encoder(images)
        if self.decoder_name == "transformer":
            return self.decoder(image_feats, captions_in, pad_mask=pad_mask)
        return self.decoder(image_feats, captions_in)
