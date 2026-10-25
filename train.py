import argparse
import torch
import torch.nn as nn
from src.model import GPT
from src.data import get_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT from scratch")

    # Model hyperparameters
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # Generation and checkpoint
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, default="gpt_shakespeare.pth")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Configuration: {args}")

    with open("data/shakespeare.txt", "r") as f:
        text = f.read()

    vocab = sorted(set(text))
    stoi = {char: i for i, char in enumerate(vocab)}
    itos = {i: char for char, i in stoi.items()}

    encoded_text = [stoi[char] for char in text]
    data = torch.tensor(encoded_text)

    train_idx = int(0.9 * len(data))
    train_data = data[:train_idx]
    val_data = data[train_idx:]

    vocab_size = len(vocab)

    # If head_size is not specified, derive it from n_embd and num_heads.
    if args.head_size is None:
        if args.n_embd % args.num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads.")
        head_size = args.n_embd // args.num_heads
    else:
        head_size = args.head_size

    gpt = GPT(
        args.num_heads,
        args.n_embd,
        head_size,
        args.block_size,
        vocab_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        gpt.parameters(),
        lr=args.learning_rate,
    )
    loss = nn.CrossEntropyLoss()

    for i in range(args.max_iter):
        if i % 10 == 0:
            gpt.eval()

            with torch.no_grad():
                x, y = get_batch(
                    val_data,
                    batch_size=args.batch_size,
                    block_size=args.block_size,
                )
                x, y = x.to(device), y.to(device)

                logits = gpt(x)
                logits = logits.view(
                    args.batch_size * args.block_size,
                    vocab_size,
                )
                y = y.view(args.batch_size * args.block_size)

                loss_value = loss(logits, y)

                print(
                    f"Validation iteration: {i}, "
                    f"val loss: {loss_value.item():.4f}"
                )

        gpt.train()

        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
        x, y = x.to(device), y.to(device)

        logits = gpt(x)
        logits = logits.view(
            args.batch_size * args.block_size,
            vocab_size,
        )
        y = y.view(args.batch_size * args.block_size)

        optimizer.zero_grad()
        loss_value = loss(logits, y)
        loss_value.backward()
        optimizer.step()

        print(f"Iteration: {i}, loss: {loss_value.item():.4f}")

    # Save model weights
    torch.save(gpt.state_dict(), args.checkpoint)
    print(f"Model saved to {args.checkpoint}")

    # Generate text
    gpt.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = gpt.generate(
            context,
            max_new_tokens=args.max_new_tokens,
        )

    result = "".join(itos[int(i)] for i in generated[0])
    print(result)