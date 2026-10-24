# GPT from scratch

A small GPT-style language model, written from scratch in PyTorch.
The model learns to predict the next character in Shakespeare's text. Then it generates new text.

## Overview

- Decoder-only Transformer. Attention, feed-forward layers, embeddings and text sampling are written by hand.
- Character-level tokenizer. The vocabulary has 65 characters.
- About 0.82 million parameters.
- Training loop with a train/validation split and text generation.

## Architecture

| Component | Choice |
| --- | --- |
| Tokens | Characters (65 symbols) |
| Embedding size | 128 |
| Layers | 4 Transformer blocks |
| Attention | 4 heads of size 32, causal mask, dropout 0.1 |
| Normalization | LayerNorm before attention and before the feed-forward layer (pre-norm) |
| Feed-forward | Linear (128 → 512), ReLU, Linear (512 → 128) |
| Position information | Learned position embeddings |
| Context length | 64 characters |
| Output | Final LayerNorm, then a linear layer to the vocabulary |

Each block applies two residual steps:

```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Generation samples the next character from the softmax of the last position. It does not use temperature or top-k. The context is cut to the last 64 characters.

## Project structure

```
.
├── data/shakespeare.txt   # training text (1,115,394 characters)
├── src/
│   ├── model.py           # Head, MultiHeadAttention, FeedForward, Transformer, GPT
│   └── data.py            # get_batch: random (input, target) windows
├── train.py               # training and generation script
├── pyproject.toml
└── uv.lock
```

## Setup

Requirements: Python 3.12 or newer, PyTorch, NumPy.

```bash
git clone https://github.com/Ilyass-42/gpt-from-scratch.git
cd gpt-from-scratch
uv sync
```

Without `uv`, install the packages with `pip install torch numpy`.

## Training

Run the script from the root of the repository:

```bash
uv run python train.py
```

The script:

1. Reads `data/shakespeare.txt` and maps each character to an integer.
2. Uses 90% of the text for training and 10% for validation.
3. Trains the model with random windows of text. The target is the input shifted by one character.
4. Prints the training loss at each iteration and the validation loss every 10 iterations.
5. Generates 200 characters at the end.

The script uses CUDA when it is available. Otherwise it uses the CPU.

| Setting | Value |
| --- | --- |
| Loss | Cross-entropy |
| Optimizer | AdamW, learning rate 1e-3 |
| Iterations | 10,000 |
| Batch size | 4 |
| Context length | 64 |
| Embedding size | 128 |
| Heads | 4 (size 32) |
| Layers | 4 |

## Roadmap

- [x] Character-level GPT baseline
- [x] Validation loop and text generation
- [x] CUDA support
- [ ] BPE tokenizer and GPT-2 baseline
- [ ] Ablations, one change at a time: RMSNorm, RoPE, SwiGLU, GQA
- [ ] Mixture of experts (MoE)
- [ ] TPU training with PyTorch/XLA
- [ ] Chat model and interface
