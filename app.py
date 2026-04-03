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

# vocab hardcoded — extracted from training dataset, no file needed at runtime
chars  = list('\n !"#$%&\'()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
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
        'what_to_notice': 'Output is often incoherent or loops back on itself. Attention can copy and reweight tokens, but without a feedforward layer there\'s no way to combine or transform that information — so the model mostly shuffles what it\'s seen.',
    },
    '20260403_012829': {
        'label': 'v2 — + FeedForward (1×)',
        'desc':  'Added a feedforward layer after attention (same width as embedding).',
        'what_to_notice': 'Slightly more varied word choices. The feedforward layer adds a per-token transformation step, but at 1× width it\'s narrow — like adding one extra neuron layer with the same bottleneck size. A small but real improvement.',
    },
    '20260403_014431': {
        'label': 'v3 — + FeedForward (4×)',
        'desc':  'Expanded feedforward to 4× the embedding width, as in the original transformer.',
        'what_to_notice': 'Noticeably more structure — words tend to cluster into metal-ish phrases. The 4× expansion (from the "Attention is All You Need" paper) gives the model room to learn richer feature combinations before projecting back down.',
    },
    '20260403_015121': {
        'label': 'v3b — same arch, different seed',
        'desc':  'Same architecture as v3, trained again (fixed seed — identical output).',
        'what_to_notice': 'Compare this directly with v3. Output should be nearly identical — this shows that architecture drives quality, not lucky random initialization. If you see big differences, that\'s a signal of high variance (undertrained model).',
    },
    '20260403_022641': {
        'label': 'v4 — Blocks + LayerNorm + Residuals + Dropout',
        'desc':  'Full transformer blocks with residual connections, layer norm, 6 layers, larger model.',
        'what_to_notice': 'The most coherent output. Residual connections let gradients flow cleanly through 6 stacked layers. Layer norm stabilizes each block\'s input distribution. Dropout reduces overfitting. This is the closest to a "real" GPT architecture.',
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
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Metal-Lyrics-GPT</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      #000;
      --surface: #0f0f0f;
      --border:  #1a1a1a;
      --red:     #cc0000;
      --red-dim: #7a0000;
      --white:   #fff;
      --muted:   #555;
      --text:    #ccc;
    }

    body {
      background: var(--bg);
      color: var(--white);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* ── landing state ── */
    #landing {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      flex: 1;
      width: 100%;
      padding: 40px 24px;
      transition: all 0.4s ease;
    }

    #landing.results-shown {
      flex: 0;
      padding: 28px 24px 20px;
      border-bottom: 1px solid var(--border);
    }

    .logo-area {
      text-align: center;
      margin-bottom: 36px;
      transition: all 0.4s ease;
    }

    #landing.results-shown .logo-area {
      margin-bottom: 16px;
    }

    .logo-icon {
      font-size: 2.8rem;
      line-height: 1;
      margin-bottom: 12px;
      display: block;
      transition: all 0.4s ease;
    }

    #landing.results-shown .logo-icon {
      font-size: 1.6rem;
      margin-bottom: 6px;
    }

    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      color: var(--white);
      transition: font-size 0.4s ease;
    }

    #landing.results-shown h1 {
      font-size: 1.2rem;
    }

    .tagline {
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 6px;
      transition: all 0.4s ease;
    }

    #landing.results-shown .tagline {
      display: none;
    }

    /* ── input area ── */
    .input-wrap {
      width: 100%;
      max-width: 640px;
      position: relative;
    }

    .input-box {
      display: flex;
      align-items: flex-end;
      gap: 0;
      background: var(--surface);
      border: 1px solid #2a2a2a;
      border-radius: 14px;
      padding: 14px 16px;
      transition: border-color 0.2s;
      width: 100%;
    }

    .input-box:focus-within {
      border-color: var(--red);
    }

    textarea {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--white);
      font-family: inherit;
      font-size: 1rem;
      line-height: 1.5;
      resize: none;
      outline: none;
      min-height: 28px;
      max-height: 160px;
      overflow-y: auto;
      padding: 0;
    }

    textarea::placeholder { color: var(--muted); }

    .send-btn {
      background: var(--red);
      border: none;
      border-radius: 8px;
      color: var(--white);
      cursor: pointer;
      width: 36px;
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      margin-left: 10px;
      transition: background 0.2s, transform 0.1s;
    }

    .send-btn:hover  { background: #e60000; }
    .send-btn:active { transform: scale(0.95); }
    .send-btn:disabled { background: #2a2a2a; cursor: not-allowed; }
    .send-btn svg { width: 18px; height: 18px; fill: currentColor; }

    .options-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      font-size: 0.75rem;
      color: var(--muted);
    }

    .options-row label { white-space: nowrap; }

    .token-input {
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--white);
      font-family: inherit;
      font-size: 0.8rem;
      padding: 4px 8px;
      width: 72px;
      text-align: center;
      outline: none;
    }

    .token-input:focus { border-color: var(--red); }

    .hint {
      font-size: 0.72rem;
      color: var(--muted);
      margin-top: 14px;
      text-align: center;
    }

    #landing.results-shown .hint { display: none; }

    /* ── results grid ── */
    #results-section {
      display: none;
      width: 100%;
      max-width: 1100px;
      padding: 28px 24px 48px;
    }

    #results-section.visible { display: block; }

    .results-label {
      font-size: 0.7rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 14px;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      transition: border-color 0.2s;
    }

    .card.loading { border-color: var(--red-dim); }
    .card.done    { border-color: #1f1f1f; }

    .card-label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--white);
      margin-bottom: 3px;
    }

    .card-desc {
      font-size: 0.68rem;
      color: var(--muted);
      margin-bottom: 6px;
      line-height: 1.5;
    }

    .card-cfg {
      font-size: 0.62rem;
      color: #333;
      margin-bottom: 12px;
      font-family: monospace;
    }

    .output {
      background: #060606;
      border: 1px solid #111;
      border-radius: 8px;
      padding: 14px;
      white-space: pre-wrap;
      font-family: monospace;
      font-size: 0.78rem;
      line-height: 1.7;
      min-height: 80px;
      color: #bbb;
    }

    .output.loading-text { color: var(--red-dim); }

    .bar {
      height: 2px;
      width: 100%;
      background: var(--border);
      border-radius: 2px;
      margin-top: 10px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      width: 0%;
      background: var(--red);
      border-radius: 2px;
      transition: width 0.3s ease;
    }

    .card.done .bar-fill { width: 100%; background: #1a4a1a; }

    /* ── notice box on cards ── */
    .card-notice {
      margin-top: 12px;
      padding: 10px 12px;
      border-left: 2px solid var(--red-dim);
      background: #0a0000;
      border-radius: 0 6px 6px 0;
      font-size: 0.7rem;
      color: #888;
      line-height: 1.6;
    }

    .card-notice strong {
      display: block;
      font-size: 0.65rem;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--red-dim);
      margin-bottom: 4px;
    }

    /* ── landing explainer ── */
    .explainer {
      width: 100%;
      max-width: 640px;
      margin-top: 28px;
      border-top: 1px solid #111;
      padding-top: 24px;
    }

    .explainer-title {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--red);
      margin-bottom: 14px;
    }

    .explainer-body {
      font-size: 0.78rem;
      color: #777;
      line-height: 1.75;
      margin-bottom: 18px;
    }

    .explainer-body strong { color: #aaa; }

    .steps {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .step {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      font-size: 0.73rem;
      color: #666;
      line-height: 1.5;
    }

    .step-num {
      background: #111;
      border: 1px solid #222;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.6rem;
      color: var(--red-dim);
      font-weight: 700;
    }

    .step b { color: #999; }

    #landing.results-shown .explainer { display: none; }

    /* ── results header bar ── */
    .results-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 8px;
    }

    .results-legend {
      font-size: 0.68rem;
      color: #444;
      max-width: 560px;
      line-height: 1.6;
    }

    @media (max-width: 600px) {
      h1 { font-size: 1.5rem; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

  <div id="landing">
    <div class="logo-area">
      <span class="logo-icon">🤘</span>
      <h1>Metal-Lyrics-GPT</h1>
      <p class="tagline">Type a metal lyric — 5 GPT versions will continue it</p>
    </div>

    <div class="input-wrap">
      <div class="input-box">
        <textarea
          id="prompt"
          rows="1"
          placeholder="Enter a metal lyric or leave empty..."
          onInput="autoResize(this)"
          onKeyDown="handleKey(event)"
        ></textarea>
        <button class="send-btn" id="btn" onclick="generateAll()" title="Generate">
          <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
        </button>
      </div>

      <div class="options-row">
        <label for="tokens">Tokens:</label>
        <input type="number" id="tokens" class="token-input" value="200" min="10" max="500">
        <span style="color:#333">|</span>
        <span>Press Enter or click ▶ to generate</span>
      </div>

      <p class="hint">Compares 5 transformer checkpoints — from attention-only to full GPT blocks</p>
    </div>

    <div class="explainer">
      <div class="explainer-title">What is this demo?</div>
      <div class="explainer-body">
        A real GPT was trained from scratch on <strong>heavy metal lyrics</strong>, rebuilt piece by piece — the same
        way the original 2017 transformer paper introduced each component. Every checkpoint below is a
        <strong>different version of the same model</strong>, with one more building block added. Same prompt, same
        training data — you can see directly how each addition changes the output quality.
      </div>
      <div class="steps">
        <div class="step"><div class="step-num">1</div><div><b>Attention only</b> — the model can look at context but can't transform it. Outputs are often repetitive or incoherent.</div></div>
        <div class="step"><div class="step-num">2</div><div><b>+ Feedforward (1×)</b> — adds a small transformation layer per token. First signs of real word patterns.</div></div>
        <div class="step"><div class="step-num">3</div><div><b>+ Feedforward (4×)</b> — wider feedforward (as in the original paper). More structured, phrase-like output.</div></div>
        <div class="step"><div class="step-num">4</div><div><b>Same arch, different seed</b> — proves architecture matters more than random init. Output should match v3.</div></div>
        <div class="step"><div class="step-num">5</div><div><b>Full GPT blocks</b> — residuals, layer norm, dropout, 6 layers. The real thing. Closest to coherent lyrics.</div></div>
      </div>
    </div>
  </div>

  <div id="results-section">
    <div class="results-header">
      <div class="results-label">5 model outputs — same prompt, different architectures</div>
      <div class="results-legend">Each card adds one component to the transformer. Read them left-to-right to see the output improve as the architecture grows.</div>
    </div>
    <div class="grid" id="grid">
      {% for run_id, meta in runs.items() %}
      <div class="card" id="card-{{ run_id }}">
        <div class="card-label">{{ meta.label }}</div>
        <div class="card-desc">{{ meta.desc }}</div>
        <div class="card-cfg">{{ meta.cfg_str }}</div>
        <div class="output" id="out-{{ run_id }}">—</div>
        <div class="bar"><div class="bar-fill" id="bar-{{ run_id }}"></div></div>
        {% if meta.what_to_notice %}
        <div class="card-notice">
          <strong>What to look for</strong>
          {{ meta.what_to_notice }}
        </div>
        {% endif %}
      </div>
      {% endfor %}
    </div>
  </div>

  <script>
    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    }

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        generateAll();
      }
    }

    async function generateAll() {
      const prompt  = document.getElementById('prompt').value;
      const tokens  = parseInt(document.getElementById('tokens').value) || 200;
      const btn     = document.getElementById('btn');
      const landing = document.getElementById('landing');
      const results = document.getElementById('results-section');

      btn.disabled = true;

      const run_ids = {{ run_ids | tojson }};

      // show results section & shrink header
      results.classList.add('visible');
      landing.classList.add('results-shown');

      // set all cards to loading state
      run_ids.forEach(id => {
        const card = document.getElementById('card-' + id);
        const out  = document.getElementById('out-' + id);
        const bar  = document.getElementById('bar-' + id);
        card.className = 'card loading';
        out.className  = 'output loading-text';
        out.textContent = 'generating...';
        bar.style.width = '30%';
      });

      // scroll to results
      results.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // fire all in parallel
      await Promise.all(run_ids.map(async (run_id) => {
        const res  = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ run_id, prompt, max_tokens: tokens })
        });
        const data = await res.json();
        const card = document.getElementById('card-' + run_id);
        const out  = document.getElementById('out-' + run_id);
        const bar  = document.getElementById('bar-' + run_id);
        out.className  = 'output';
        out.textContent = data.output || ('Error: ' + data.error);
        card.className  = 'card done';
        bar.style.width = '100%';
      }));

      btn.disabled = false;
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
            runs_display[run_id] = {**meta, 'cfg_str': cfg_str, 'what_to_notice': meta.get('what_to_notice', '')}
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
