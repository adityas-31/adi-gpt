import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import sys

# ---- point this to your run folder ----
RUN_DIR = 'runs/20260403_022641'
MAX_TOKENS = 500
PROMPT = ''   # leave empty to start from nothing, or type something
# ---------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load config
with open(os.path.join(RUN_DIR, 'config.json')) as f:
    cfg = json.load(f)

block_size     = cfg['block_size']
n_embd         = cfg['n_embd']
n_heads        = cfg['n_heads']
vocabulary_size = cfg['vocabulary_size']
n_layer        = cfg.get('n_layer', 3)
dropout        = cfg.get('dropout', 0.0)

# rebuild vocab from dataset (same as training)
with open('dataset/input.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# model definition (must match train.py exactly)
class Head(nn.Module):
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
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*(Block(n_embd, n_head=n_heads) for _ in range(n_layer)))
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x), None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# load weights
m = BigramLanguageModel(vocabulary_size).to(device)
m.load_state_dict(torch.load(os.path.join(RUN_DIR, 'weights.pt'), map_location=device))
m.eval()

# encode prompt (or start from newline if empty)
if PROMPT:
    idx = torch.tensor(encode(PROMPT), dtype=torch.long, device=device).unsqueeze(0)
else:
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

print(f"--- Generating {MAX_TOKENS} tokens from: '{PROMPT or '<newline>'}' ---\n")
output = decode(m.generate(idx, max_new_tokens=MAX_TOKENS)[0].tolist())
print(output)
