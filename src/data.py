import torch
import torch.nn as nn


def get_batch(data, batch_size, block_size):
    x = torch.zeros(batch_size, block_size, dtype=torch.long)
    y = torch.zeros(batch_size, block_size, dtype=torch.long)

    for i in range(batch_size):
        j = torch.randint(0, len(data)-block_size, (1,)).item()
        x[i,:] = data[j:j+block_size]
        y[i,:] = data[j+1:j+block_size+1]

    return x,y
            
    
if __name__ == "__main__":
    with open("data/shakespeare.txt", "r") as f:
        text = f.read()
    vocab = sorted(set(text))

    stoi = {char : i for i,char in enumerate(vocab)}
    itos = {i: char for char,i in stoi.items() }

    encoded_text  = [stoi[char] for char in text]

    data = torch.tensor(encoded_text)

    train_idx = int(0.9*len(data))
    val_idx = len(data) - train_idx

    train_data = data[:train_idx]
    val_data = data[train_idx:]



