import torch
import torch.nn as nn
from torch.nn import functional as F
from flask import Flask, request, jsonify, render_template_string
import json
import os
import threading
import time
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# rebuild vocab
with open('dataset/input.txt', 'r') as f:
    text = f.read()
chars  = sorted(list(set(text)))
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])


# ── model building blocks ──────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
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
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout=0.0, has_proj=True):
        super().__init__()
        head_size    = n_embd // num_heads
        self.heads   = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd) if has_proj else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out) if self.proj else out


class FeedForward(nn.Module):
    def __init__(self, n_embd, expand=1, dropout=0.0, single_linear=False):
        super().__init__()
        if single_linear:
            self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embd, expand * n_embd),
                nn.ReLU(),
                nn.Linear(expand * n_embd, n_embd),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.sa   = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, expand=4, dropout=dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ── flexible model that covers all 4 architectures ────────────────────────

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, block_size,
                 n_layer=0, ffwd_expand=0, dropout=0.0, has_proj=True, single_linear=False):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # block-based architecture
        if n_layer > 0:
            self.blocks  = nn.Sequential(*(Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)))
            self.ln_f    = nn.LayerNorm(n_embd)
            self.sa_heads = None
            self.ffwd     = None
        else:
            self.blocks  = None
            self.ln_f    = None
            self.sa_heads = MultiHeadAttention(n_embd, n_heads, block_size, dropout, has_proj=has_proj)
            self.ffwd     = FeedForward(n_embd, expand=ffwd_expand, dropout=dropout, single_linear=single_linear) if ffwd_expand > 0 else None

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        if self.blocks is not None:
            x = self.ln_f(self.blocks(x))
        else:
            x = self.sa_heads(x)
            if self.ffwd:
                x = self.ffwd(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits   = self(idx_cond)
            probs    = F.softmax(logits[:, -1, :], dim=-1)
            idx      = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# ── auto-detect architecture and load a run ───────────────────────────────

def load_run(run_dir):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)

    sd          = torch.load(os.path.join(run_dir, 'weights.pt'), map_location=device, weights_only=True)
    n_embd      = cfg['n_embd']
    n_heads     = cfg['n_heads']
    block_size  = cfg['block_size']
    vocab_size  = cfg['vocabulary_size']
    dropout     = cfg.get('dropout', 0.0)

    has_blocks = any(k.startswith('blocks.') for k in sd)
    has_ffwd   = 'ffwd.net.0.weight' in sd

    has_proj          = 'sa_heads.proj.weight' in sd
    has_second_linear = 'ffwd.net.2.weight' in sd

    if has_blocks:
        n_layer       = cfg.get('n_layer', 6)
        ffwd_expand   = 0
        single_linear = False
    elif has_ffwd:
        ffwd_out      = sd['ffwd.net.0.weight'].shape[0]
        ffwd_expand   = ffwd_out // n_embd   # 1 or 4
        single_linear = not has_second_linear
        n_layer       = 0
    else:
        ffwd_expand   = 0
        single_linear = False
        n_layer       = 0

    model = GPT(vocab_size, n_embd, n_heads, block_size, n_layer, ffwd_expand, dropout, has_proj, single_linear).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


# ── run metadata ──────────────────────────────────────────────────────────

RUNS = {
    '20260329_011932': {
        'label': 'v1 — Attention only',
        'desc':  'Token + position embeddings, multi-head self attention, no feedforward.',
    },
    '20260403_012829': {
        'label': 'v2 — + FeedForward (1×)',
        'desc':  'Added a feedforward layer after attention (same width as embedding).',
    },
    '20260403_014431': {
        'label': 'v3 — + FeedForward (4×)',
        'desc':  'Expanded feedforward to 4× the embedding width, as in the original transformer.',
    },
    '20260403_015121': {
        'label': 'v3b — same arch, different seed',
        'desc':  'Same architecture as v3, trained again (fixed seed — identical output).',
    },
    '20260403_022641': {
        'label': 'v4 — Blocks + LayerNorm + Residuals + Dropout',
        'desc':  'Full transformer blocks with residual connections, layer norm, 6 layers, larger model.',
    },
}

print("Loading models...")
MODELS = {}
for run_id, meta in RUNS.items():
    run_dir = os.path.join('runs', run_id)
    if os.path.exists(run_dir):
        MODELS[run_id] = load_run(run_dir)
        print(f"  ✓ {meta['label']}")
print("All models loaded.\n")


# ── flask app ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024  # 8KB max request body

inference_lock = threading.Lock()

# simple rate limiter — max 10 requests per IP per minute
_rate_data = defaultdict(list)
_rate_lock = threading.Lock()
MAX_REQUESTS_PER_MINUTE = 10

def is_rate_limited(ip):
    now = time.time()
    with _rate_lock:
        _rate_data[ip] = [t for t in _rate_data[ip] if now - t < 60]
        if len(_rate_data[ip]) >= MAX_REQUESTS_PER_MINUTE:
            return True
        _rate_data[ip].append(now)
    return False

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>adi-gpt — model comparison</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #ddd; font-family: monospace; padding: 40px 24px; }
    h1   { font-size: 1.3rem; color: #fff; margin-bottom: 4px; }
    .sub { font-size: 0.75rem; color: #555; margin-bottom: 36px; }

    .controls { display: flex; gap: 16px; align-items: flex-end; margin-bottom: 32px; flex-wrap: wrap; }
    .field label { display: block; font-size: 0.7rem; color: #666; margin-bottom: 5px; }
    textarea, input[type=number] {
      background: #161616; border: 1px solid #222; color: #ddd;
      font-family: monospace; font-size: 0.85rem; padding: 10px 12px;
      border-radius: 5px; outline: none;
    }
    textarea { width: 420px; height: 80px; resize: vertical; }
    input[type=number] { width: 100px; }
    button {
      background: #fff; color: #000; border: none; padding: 10px 22px;
      font-family: monospace; font-size: 0.85rem; border-radius: 5px;
      cursor: pointer; white-space: nowrap;
    }
    button:disabled { background: #333; color: #666; cursor: not-allowed; }

    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
    .card { background: #141414; border: 1px solid #1f1f1f; border-radius: 8px; padding: 20px; }
    .card-header { margin-bottom: 12px; }
    .card-label { font-size: 0.85rem; color: #fff; font-weight: bold; margin-bottom: 4px; }
    .card-desc  { font-size: 0.7rem; color: #555; margin-bottom: 8px; line-height: 1.5; }
    .card-cfg   { font-size: 0.65rem; color: #444; }
    .output {
      background: #0f0f0f; border: 1px solid #1a1a1a; border-radius: 5px;
      padding: 14px; white-space: pre-wrap; font-size: 0.8rem; line-height: 1.65;
      min-height: 60px; color: #bbb; margin-top: 12px;
    }
    .spinner { color: #444; font-size: 0.75rem; margin-top: 12px; }
  </style>
</head>
<body>
  <h1>adi-gpt</h1>
  <p class="sub">each card is a checkpoint from the tutorial — same prompt, different architecture</p>

  <div class="controls">
    <div class="field">
      <label>Prompt (leave empty for newline start)</label>
      <textarea id="prompt" placeholder="To be or not to be..."></textarea>
    </div>
    <div class="field">
      <label>Tokens</label>
      <input type="number" id="tokens" value="200" min="10" max="500">
    </div>
    <button id="btn" onclick="generateAll()">Generate All</button>
  </div>

  <div class="grid" id="grid">
    {% for run_id, meta in runs.items() %}
    <div class="card" id="card-{{ run_id }}">
      <div class="card-header">
        <div class="card-label">{{ meta.label }}</div>
        <div class="card-desc">{{ meta.desc }}</div>
        <div class="card-cfg">{{ meta.cfg_str }}</div>
      </div>
      <div class="output" id="out-{{ run_id }}">—</div>
    </div>
    {% endfor %}
  </div>

  <script>
    async function generateAll() {
      const prompt = document.getElementById('prompt').value;
      const tokens = parseInt(document.getElementById('tokens').value);
      const btn    = document.getElementById('btn');
      btn.disabled = true;
      btn.textContent = 'Generating...';

      const run_ids = {{ run_ids | tojson }};

      // set all cards to loading
      run_ids.forEach(id => {
        document.getElementById('out-' + id).textContent = 'generating...';
      });

      // fire all requests in parallel
      await Promise.all(run_ids.map(async (run_id) => {
        const res  = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ run_id, prompt, max_tokens: tokens })
        });
        const data = await res.json();
        document.getElementById('out-' + run_id).textContent = data.output || data.error;
      }));

      btn.disabled    = false;
      btn.textContent = 'Generate All';
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    runs_display = {}
    for run_id, meta in RUNS.items():
        if run_id in MODELS:
            _, cfg = MODELS[run_id]
            cfg_str = f"n_embd={cfg['n_embd']} · n_heads={cfg['n_heads']} · block_size={cfg['block_size']} · iters={cfg['max_iters']}"
            runs_display[run_id] = {**meta, 'cfg_str': cfg_str}
    return render_template_string(HTML, runs=runs_display, run_ids=list(runs_display.keys()))


@app.route('/generate', methods=['POST'])
def generate():
    if is_rate_limited(request.remote_addr):
        return jsonify({'error': 'too many requests — slow down'}), 429

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'invalid request'}), 400

    run_id = data.get('run_id', '')
    if run_id not in MODELS:
        return jsonify({'error': 'model not found'}), 400

    prompt = str(data.get('prompt', '')).strip()
    if len(prompt) > 500:
        return jsonify({'error': 'prompt too long (max 500 chars)'}), 400

    try:
        max_tokens = int(data.get('max_tokens', 200))
        max_tokens = max(10, min(max_tokens, 500))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid max_tokens'}), 400

    model, _ = MODELS[run_id]

    with inference_lock:
        with torch.no_grad():
            if prompt:
                idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            else:
                idx = torch.zeros((1, 1), dtype=torch.long, device=device)

            out    = model.generate(idx, max_new_tokens=max_tokens)
            result = decode(out[0].tolist())
            if prompt:
                result = result[len(prompt):]

    return jsonify({'output': result})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
