import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

n_embd = 32
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

assert torch.cuda.is_available(), "CUDA not available! Check your PyTorch install."
device = torch.device('cuda')
print(f"Using device: {torch.cuda.get_device_name(0)}")

with open('dataset/input.txt' , 'r') as file:
    text = file.read()
    print(str(len(text)) + " characters long")


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd , head_size, bias=False)
        self.query = nn.Linear(n_embd , head_size , bias=False)
        self.value = nn.Linear(n_embd , head_size , bias=False)
        self.register_buffer('tril' , torch.tril(torch.ones(block_size , block_size)))

    def forward(self , x):
        B , T , C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        #compute attention scores / affinities

        wei = q @ k.transpose(-2 , -1) * C**-0.5 # (B , T , C) @ (B , C , T) -> (B , T , T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float('-inf')) #(B,T,T)
        wei = F.softmax (wei , dim=-1) # (B , T , T)
        
        #weighyted aggregation of values
        v = self.value(x) #(B , T , C)
        out = wei @ v #(B , T , T) @ (B , T , C) -> (B , T , C)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads] , dim=-1)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocabulary_size):
        super().__init__()
        #each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size , n_embd) #(B , T , C)
        self.position_embedding_table = nn.Embedding(block_size , n_embd)
        self.sa_heads = MultiHeadAttention(4 , n_embd//4)
        self.lm_head = nn.Linear(n_embd , vocabulary_size)
    def forward(self, idx, targets=None):
        B , T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embds = self.token_embedding_table(idx) # (B,T,C)
        pos_embds = self.position_embedding_table(torch.arange(T , device=device))
        x = token_embds + pos_embds
        # x = self.sa_head(x) # apply one head of self attention (B , T , C)
        x = self.sa_heads(x)
        logits = self.lm_head(x) #(B , T , vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


#this part basically takes "chunks" of the dataset in x and then the target value of the chunk is in y - for example the first 3 in x have the target as the 3rd in y. 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_dataset if split == 'train' else test_dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



# print(text[:1000])

chars = sorted(list(set(text)))
vocabulary_size = len(chars)

print("Characters are - " + ''.join(chars))
print("Vocabulary Size is " + str(vocabulary_size))


stoi = {ch:i for i ,ch in enumerate(chars)}
itos = {i:ch for i , ch in enumerate(chars)}

#Simple encoder and decoder function - string to integer and integer to string

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#encode the dataset into a tensor
data = torch.tensor(encode(text) , dtype=torch.long)
print(data.shape , data.dtype)
print(data[:1000])

#split dataset into train and vallidate
n = int(0.9*len(data))
train_dataset = data[:n]
test_dataset = data[n:]

torch.manual_seed(1337)

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')


print(xb)



m = BigramLanguageModel(vocabulary_size).to(device)
logits , loss = m(xb , yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))