# Image Captioning System (CNN Encoder + LSTM/Transformer Decoder)

This project trains an image caption generator using:

- A frozen pretrained CNN encoder (`resnet50` or `vgg16`)
- A text decoder (`lstm` or `transformer`)
- A custom vocabulary built from `captions.csv`

## Project files

- `config.py` - default configuration values
- `dataset.py` - tokenization, vocabulary, dataset, collate function
- `models.py` - encoder and decoder model definitions
- `train.py` - model training and checkpoint saving
- `inference.py` - caption generation for one image
- `requirements.txt` - dependencies

## Dataset format expected by code

`captions_file` must be a CSV with columns:

- `image`: image filename (example: `cat1.jpg`)
- `caption`: caption text

Images are read from `images_dir`, and `image` values are joined to that path.

Default paths:

- images directory: `data/images`
- captions file: `data/captions.csv`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

Basic run (uses defaults from `config.py`):

```bash
python train.py
```

Example with explicit options:

```bash
python train.py --images_dir data/images --captions_file data/captions.csv --encoder resnet50 --decoder lstm --epochs 10 --batch_size 32
```

Important train arguments:

- `--encoder`: `resnet50` or `vgg16`
- `--decoder`: `lstm` or `transformer`
- `--max_len`: caption length limit (default 30)
- `--min_word_freq`: minimum token frequency for vocabulary (default 2)

Training output:

- `artifacts/model.pt` (checkpoint with model weights + metadata)
- `artifacts/vocab.json` (saved vocabulary)

## Generate caption (inference)

```bash
python inference.py --image_path data/images/sample.jpg --checkpoint artifacts/model.pt --vocab_path artifacts/vocab.json --encoder resnet50 --decoder lstm --max_len 30
```

Notes:

- Use the same encoder/decoder combination used during training.
- Inference performs greedy decoding token by token until `<eos>` or max length.

## Dependency list

- `torch`
- `torchvision`
- `pillow`
- `pandas`
- `tqdm`
