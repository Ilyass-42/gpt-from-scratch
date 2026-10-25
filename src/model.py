import math
import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q,k.transpose(-2,-1))
        scores = scores/math.sqrt(self.key.out_features)

        T = x.shape[1]

        scores.masked_fill_(self.mask[:T, :T]==0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)

        weights =self.dropout(weights)

        out = torch.matmul(weights, v)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, block_size, head_size, num_heads, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout) # 10% of the activations are set to zero during training
        # reduces overfitting
    def forward(self, x):
        outs = [head(x) for head in self.heads]
        out = torch.cat(outs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module): 
    """transforms each tokens tensors to learn new features"""
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd,4*n_embd) # to increase the dimension of out tensor
        self.relu = nn.ReLU() # so that the attention output learns non linear relations
        self.linear_2 =  nn.Linear(4*n_embd, n_embd) # to reduce the dimension to its initial dimension

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        
        return out

class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.eps = eps
    def forward(self, x):
        return (x/ torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps))*self.gamma



class Transformer(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.MultiHeadAttention = MultiHeadAttention(block_size=block_size,
                                                     n_embd=n_embd,
                                                     head_size=head_size,
                                                     num_heads=num_heads,
                                                     dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd=n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer_norm_1(x)
        out = self.MultiHeadAttention(out)

        out_1 = out + x

        out_1_norm = self.layer_norm_2(out_1)

        out = self.mlp(out_1_norm)
        out = self.dropout(out)

        out_2 = out_1 + out
        
        return out_2


class GPT(nn.Module):
    def __init__(self,num_heads, n_embd, head_size, block_size, vocab_size,dropout=0.1):
        super().__init__()
        self.positional_encoding = nn.Embedding(block_size, n_embd) 
        self.token_embedding = nn.Embedding(vocab_size, n_embd) # token ---> embedding
        self.blocs = nn.Sequential(*[
            Transformer(num_heads, n_embd, head_size, block_size, dropout=dropout)
            for _ in range(4) # 4 transformers 
            ])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd,vocab_size)

    def forward(self,x):
        T = x.shape[1]
        positions = torch.arange(0,T, device=x.device)
        x = self.token_embedding(x)  + self.positional_encoding(positions)
        out = self.blocs(x)
        out = self.layer_norm(out)
        out = self.linear(out)
        return out

    def generate(self,x, max_new_tokens):
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.positional_encoding.num_embeddings:]
            logits = self.forward(x_cond)
            last_logit = logits[:,-1,:]
            proba = torch.softmax(last_logit,dim=-1)
            next_char = torch.multinomial(proba, num_samples=1)
            x = torch.cat([x, next_char], dim=1)
        return x


if __name__ == "__main__":
    n_embd = 128
    num_heads = 4
    head_size = 32
    block_size = 8
    vocab_size = 10
    
    x = torch.randint(0, vocab_size, (4,6))

    gpt = GPT(num_heads, n_embd, head_size, block_size,vocab_size)
    out = gpt(x)
    print(out.shape)