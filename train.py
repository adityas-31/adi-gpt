import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
from datetime import datetime

# hyperparameters
batch_size = 16
block_size = 128
n_embd = 384
n_heads = 6
n_layer = 6
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
dropout = 0.2
max_output_tokens = 10000
# ------------

assert torch.cuda.is_available(), "CUDA not available! Check your PyTorch install."
device = torch.device('cuda')
torch.manual_seed(1337)

# data
with open('dataset/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocabulary_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_dataset = data[:n]
test_dataset = data[n:]


def get_batch(split):
    data = train_dataset if split == 'train' else test_dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
#Below is a linear layer then followed bya non linear layer
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd , 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd , n_embd),
            nn.Dropout(dropout)
        )

    def forward(self , x):
        return self.net(x)

class Block(nn.Module):
    """transformer block - communication followed by computation"""

    def __init__(self, n_embd , n_head):
        #n_embd = embedding dimension , n_head = number of heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head , head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #this is the communication
        x = x+ self.ffwd(self.ln2(x)) #this is the computation
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(n_heads, n_embd // n_heads)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*(Block(n_embd , n_head=n_heads) for _ in range(n_layer)))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embds = self.token_embedding_table(idx)
        pos_embds   = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embds + pos_embds
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# training
m = BigramLanguageModel(vocabulary_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(f"Training on {torch.cuda.get_device_name(0)}")
print(f"Vocab size: {vocabulary_size} | params: {sum(p.numel() for p in m.parameters()):,}")

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:5d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate output
generated = decode(m.generate(
    idx=torch.zeros((1, 1), dtype=torch.long, device=device),
    max_new_tokens=max_output_tokens
)[0].tolist())

print("\n--- Generated Output ---")
print(generated)

# save run
run_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(run_dir)

torch.save(m.state_dict(), os.path.join(run_dir, 'weights.pt'))

with open(os.path.join(run_dir, 'output.txt'), 'w') as f:
    f.write(generated)

with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    json.dump({
        'batch_size': batch_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_heads': n_heads,
        'max_iters': max_iters,
        'learning_rate': learning_rate,
        'vocabulary_size': vocabulary_size,
        'n_layer': n_layer
    }, f, indent=2)

print(f"\nRun saved to {run_dir}/")
