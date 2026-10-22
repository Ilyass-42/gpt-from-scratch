import torch
import torch.nn as nn
from src.model import GPT
from src.data import get_batch

if __name__=="__main__":
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

    batch_size=4

    
    n_embd = 128
    num_heads = 4
    head_size = 32
    block_size = 64
    max_iter = 100
    vocab_size = len(vocab)

    gpt = GPT(num_heads,n_embd,head_size, block_size, vocab_size)


    optimizer = torch.optim.AdamW(gpt.parameters(), lr = 1e-3)
    loss = nn.CrossEntropyLoss()

    for i in range(max_iter):
        if i % 10 == 0:
            gpt.eval() # disables dropout
            with torch.no_grad():
                x, y = get_batch(val_data, batch_size=batch_size, block_size=8)
                logits = gpt(x) # 4, 8, 65
                logits = logits.view(batch_size*block_size, vocab_size)
                y_ = y.view(batch_size*block_size)
                loss_value = loss(logits, y_)
                print(f"Validation iteration : {i}, val loss : {loss_value.item():.4f}")
        gpt.train() # enables the dropout

        x, y = get_batch(train_data, batch_size=batch_size, block_size=8)

        logits = gpt(x) # 4, 8, 65
        logits = logits.view(batch_size*block_size, vocab_size)
        
        y_ = y.view(batch_size*block_size)
        
        optimizer.zero_grad()

        loss_value = loss(logits,y_)
        loss_value.backward()    
        optimizer.step()
        
        print(f"iteration : {i}, loss : {loss_value.item():.4f}")
        


