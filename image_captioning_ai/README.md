# Image Captioning AI (CNN + NLP)

This project combines:
- **Computer Vision**: pre-trained `ResNet50` or `VGG16` to extract image features
- **Natural Language Processing**: a caption generator using either:
  - `LSTM` decoder (RNN-based), or
  - `Transformer` decoder

## Features

- Encoder choices: `resnet50`, `vgg16`
- Decoder choices: `lstm`, `transformer`
- Train on caption datasets in a simple CSV format
- Generate captions for new images after training

## Project Structure

- `config.py` - hyperparameters and defaults
- `dataset.py` - vocabulary, dataset, and dataloader
- `models.py` - encoder and decoder architectures
- `train.py` - training loop
- `inference.py` - caption generation for a single image
- `requirements.txt` - Python dependencies

## Dataset Format

Use a CSV file with two columns:

- `image`: image filename (example: `dog_001.jpg`)
- `caption`: text caption (example: `a brown dog is running`)

Images should be in a single directory, for example:

```
data/
  images/
    dog_001.jpg
    cat_010.jpg
  captions.csv
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Example (ResNet + LSTM):

```bash
python train.py --images_dir data/images --captions_file data/captions.csv --encoder resnet50 --decoder lstm
```

Example (VGG + Transformer):

```bash
python train.py --images_dir data/images --captions_file data/captions.csv --encoder vgg16 --decoder transformer
```

Saved artifacts:
- `artifacts/model.pt`
- `artifacts/vocab.json`

## Inference

```bash
python inference.py --image_path data/images/dog_001.jpg --checkpoint artifacts/model.pt --vocab_path artifacts/vocab.json --encoder resnet50 --decoder lstm
```

## Notes

- Start with LSTM if you are new to captioning pipelines.
- Transformer decoder often benefits from more training data.
- For better results, use datasets like Flickr8k/Flickr30k/MS-COCO after adapting captions into the required CSV format.
