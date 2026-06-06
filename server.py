"""
server.py  –  FAQ Chatbot using Python's built-in http.server (NO Flask needed)
Run:  python server.py
Open: http://localhost:8000
"""

import json, re, math, os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Download NLTK data ────────────────────────────────────────────────────────
import nltk
for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(res, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

# ── NLP helpers ───────────────────────────────────────────────────────────────

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 1]
    return tokens

def build_vocab(corpus_tokens):
    vocab = {}
    idx   = 0
    for tokens in corpus_tokens:
        for t in tokens:
            if t not in vocab:
                vocab[t] = idx
                idx += 1
    return vocab

def tf(tokens):
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens) or 1
    return {t: c/total for t, c in freq.items()}

def idf(vocab, corpus_tokens):
    N      = len(corpus_tokens)
    scores = {}
    for term in vocab:
        df = sum(1 for doc in corpus_tokens if term in doc)
        scores[term] = math.log((N + 1) / (df + 1)) + 1
    return scores

def tfidf_vector(tokens, vocab, idf_scores):
    tf_scores = tf(tokens)
    vec = [0.0] * len(vocab)
    for term, tidx in vocab.items():
        if term in tf_scores:
            vec[tidx] = tf_scores[term] * idf_scores.get(term, 1)
    return vec

def cosine_similarity(v1, v2):
    dot  = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

# ── FAQ Engine ────────────────────────────────────────────────────────────────

class FAQChatbot:
    THRESHOLD = 0.10

    def __init__(self, path='faqs.json'):
        with open(path, encoding='utf-8') as f:
            self.faqs = json.load(f)

        self.questions = [item['question'] for item in self.faqs]
        self.answers   = [item['answer']   for item in self.faqs]

        # Build TF-IDF from scratch (no sklearn)
        self.corpus_tokens = [preprocess(q) for q in self.questions]
        self.vocab         = build_vocab(self.corpus_tokens)
        self.idf_scores    = idf(self.vocab, self.corpus_tokens)
        self.doc_vectors   = [
            tfidf_vector(tokens, self.vocab, self.idf_scores)
            for tokens in self.corpus_tokens
        ]

    def get_answer(self, query):
        query = query.strip()
        if not query:
            return {'answer': 'Please type a question!',
                    'matched_question': None, 'confidence': 0, 'status': 'empty'}

        q_tokens = preprocess(query)
        if not q_tokens:
            return {'answer': 'Could you rephrase that?',
                    'matched_question': None, 'confidence': 0, 'status': 'low'}

        q_vec = tfidf_vector(q_tokens, self.vocab, self.idf_scores)
        sims  = [cosine_similarity(q_vec, dv) for dv in self.doc_vectors]
        best  = max(range(len(sims)), key=lambda i: sims[i])
        conf  = round(sims[best], 4)

        if conf < self.THRESHOLD:
            return {
                'answer': "I don't have an answer for that. Please contact support@example.com.",
                'matched_question': None,
                'confidence': conf,
                'status': 'low_confidence'
            }

        return {
            'answer': self.answers[best],
            'matched_question': self.questions[best],
            'confidence': conf,
            'status': 'match'
        }

# ── HTTP Server ───────────────────────────────────────────────────────────────

BOT = FAQChatbot('faqs.json')

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FAQ Chatbot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;overflow:hidden}
.sidebar{width:270px;min-width:270px;background:#161b22;border-right:1px solid #21262d;display:flex;flex-direction:column;overflow:hidden}
.brand{padding:20px 16px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:10px}
.brand-icon{width:38px;height:38px;background:linear-gradient(135deg,#58a6ff,#a371f7);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px}
.brand-text h2{font-size:15px;font-weight:700}
.brand-text p{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
.sidebar-label{padding:14px 16px 6px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8b949e;font-weight:600}
.faq-list{overflow-y:auto;flex:1;padding:0 8px 8px}
.faq-list::-webkit-scrollbar{width:3px}
.faq-list::-webkit-scrollbar-thumb{background:#21262d;border-radius:3px}
.faq-btn{width:100%;text-align:left;background:none;border:none;color:#8b949e;padding:9px 10px;border-radius:6px;font-size:12px;cursor:pointer;line-height:1.5;transition:.15s}
.faq-btn:hover{background:#21262d;color:#e6edf3}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.chat-header{padding:16px 24px;border-bottom:1px solid #21262d;background:#161b22;display:flex;align-items:center;justify-content:space-between}
.chat-header h1{font-size:15px;font-weight:600}
.chat-header p{font-size:11px;color:#8b949e;margin-top:2px}
.online{display:flex;align-items:center;gap:6px;font-size:12px;color:#3fb950}
.online::before{content:'';width:8px;height:8px;background:#3fb950;border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.chat-body{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:14px}
.chat-body::-webkit-scrollbar{width:4px}
.chat-body::-webkit-scrollbar-thumb{background:#21262d;border-radius:4px}
.welcome{background:#161b22;border:1px solid #21262d;border-radius:14px;padding:28px;text-align:center;margin:auto;max-width:380px}
.welcome .wi{font-size:42px;margin-bottom:12px}
.welcome h2{font-size:18px;font-weight:700;margin-bottom:8px}
.welcome p{font-size:13px;color:#8b949e;line-height:1.7}
.msg{display:flex;gap:10px;animation:fadeUp .25s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg.user{flex-direction:row-reverse}
.av{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.av.bot{background:linear-gradient(135deg,#58a6ff,#a371f7)}
.av.user{background:#1f3a5f;border:1px solid #58a6ff}
.bubble{max-width:66%;padding:11px 15px;border-radius:12px;font-size:13.5px;line-height:1.65}
.msg.bot .bubble{background:#161b22;border:1px solid #21262d;border-bottom-left-radius:3px}
.msg.user .bubble{background:#1f3a5f;border:1px solid #2d5a8e;border-bottom-right-radius:3px}
.meta{margin-top:7px;font-size:11px;color:#8b949e;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.cbar{display:inline-flex;align-items:center;gap:4px}
.cbar-bg{width:55px;height:3px;background:#21262d;border-radius:2px;overflow:hidden}
.cbar-fill{height:100%;border-radius:2px;transition:width .6s}
.typing{display:none}
.typing.show{display:flex}
.dots{display:flex;align-items:center;gap:4px;padding:13px 16px;background:#161b22;border:1px solid #21262d;border-radius:12px;border-bottom-left-radius:3px}
.d{width:7px;height:7px;background:#8b949e;border-radius:50%;animation:bop 1.2s infinite}
.d:nth-child(2){animation-delay:.2s}
.d:nth-child(3){animation-delay:.4s}
@keyframes bop{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px);background:#58a6ff}}
.input-area{padding:14px 24px 18px;border-top:1px solid #21262d;background:#161b22}
.input-row{display:flex;gap:8px;align-items:flex-end;background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:9px 10px;transition:.2s}
.input-row:focus-within{border-color:#58a6ff}
textarea{flex:1;background:none;border:none;outline:none;font-family:inherit;font-size:13.5px;color:#e6edf3;resize:none;max-height:110px;line-height:1.5}
textarea::placeholder{color:#8b949e}
.send{width:36px;height:36px;background:linear-gradient(135deg,#58a6ff,#a371f7);border:none;border-radius:8px;cursor:pointer;color:#fff;font-size:16px;display:flex;align-items:center;justify-content:center;transition:.15s;flex-shrink:0}
.send:hover{opacity:.85}
.send:active{transform:scale(.93)}
.send:disabled{opacity:.35;cursor:not-allowed}
.hint{text-align:center;font-size:11px;color:#8b949e;margin-top:7px}
</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">
    <div class="brand-icon">🤖</div>
    <div class="brand-text"><h2>FAQ Bot</h2><p>NLP Assistant</p></div>
  </div>
  <div class="sidebar-label">Suggested Questions</div>
  <div class="faq-list" id="faqList"></div>
</aside>

<main class="main">
  <header class="chat-header">
    <div><h1>Customer Support Assistant</h1><p>TF-IDF + Cosine Similarity • NLTK Preprocessing</p></div>
    <div class="online">Online</div>
  </header>

  <div class="chat-body" id="chatBody">
    <div class="welcome" id="welcome">
      <div class="wi">💬</div>
      <h2>How can I help you?</h2>
      <p>Ask me anything about orders, payments, returns, or your account. Click a question on the left to start!</p>
    </div>
  </div>

  <div class="msg bot typing" id="typing">
    <div class="av bot">🤖</div>
    <div class="dots"><div class="d"></div><div class="d"></div><div class="d"></div></div>
  </div>

  <div class="input-area">
    <div class="input-row">
      <textarea id="inp" rows="1" placeholder="Type your question here…" maxlength="400"></textarea>
      <button class="send" id="sendBtn" onclick="send()">➤</button>
    </div>
    <div class="hint">Enter to send • Shift+Enter for new line</div>
  </div>
</main>

<script>
const chatBody = document.getElementById('chatBody');
const inp      = document.getElementById('inp');
const sendBtn  = document.getElementById('sendBtn');
const typing   = document.getElementById('typing');
const welcome  = document.getElementById('welcome');
const faqList  = document.getElementById('faqList');

// Load FAQ questions for sidebar
fetch('/faqs')
  .then(r => r.json())
  .then(data => {
    data.forEach(item => {
      const btn = document.createElement('button');
      btn.className   = 'faq-btn';
      btn.textContent = item.question;
      btn.onclick     = () => { inp.value = item.question; send(); };
      faqList.appendChild(btn);
    });
  });

inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 110) + 'px';
});

function addMsg(text, role, meta) {
  if (welcome) welcome.style.display = 'none';
  const wrap   = document.createElement('div');
  wrap.className = 'msg ' + role;
  const av   = document.createElement('div');
  av.className = 'av ' + role;
  av.textContent = role === 'bot' ? '🤖' : '👤';
  const bub  = document.createElement('div');
  bub.className  = 'bubble';
  bub.textContent = text;
  if (meta && meta.matched_question) {
    const pct   = Math.round(meta.confidence * 100);
    const color = pct >= 70 ? '#3fb950' : pct >= 40 ? '#d29922' : '#f85149';
    const m     = document.createElement('div');
    m.className = 'meta';
    m.innerHTML = `✓ <em>${meta.matched_question}</em>
      <span class="cbar"><span class="cbar-bg">
        <span class="cbar-fill" style="width:${pct}%;background:${color}"></span>
      </span>${pct}%</span>`;
    bub.appendChild(m);
  }
  wrap.appendChild(av);
  wrap.appendChild(bub);
  chatBody.appendChild(wrap);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function showTyping(v) {
  typing.classList.toggle('show', v);
  typing.style.display = v ? 'flex' : 'none';
  chatBody.scrollTop = chatBody.scrollHeight;
}

async function send() {
  const text = inp.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  inp.value = ''; inp.style.height = 'auto';
  sendBtn.disabled = true;
  showTyping(true);
  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    showTyping(false);
    addMsg(data.answer, 'bot', data);
  } catch {
    showTyping(false);
    addMsg('Connection error. Make sure server.py is running.', 'bot');
  }
  sendBtn.disabled = false;
  inp.focus();
}
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  {args[0]} {args[1]}")   # clean minimal log

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._respond(200, 'text/html', HTML.encode())

        elif path == '/faqs':
            body = json.dumps(BOT.faqs).encode()
            self._respond(200, 'application/json', body)

        else:
            self._respond(404, 'text/plain', b'Not found')

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/chat':
            length = int(self.headers.get('Content-Length', 0))
            raw    = self.rfile.read(length)
            try:
                data   = json.loads(raw)
                result = BOT.get_answer(data.get('message', ''))
            except Exception as e:
                result = {'answer': str(e), 'status': 'error'}
            body = json.dumps(result).encode()
            self._respond(200, 'application/json', body)
        else:
            self._respond(404, 'text/plain', b'Not found')

    def _respond(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    PORT   = 8000
    server = HTTPServer(('localhost', PORT), Handler)
    print(f"\n✅  FAQ Chatbot running at  http://localhost:{PORT}")
    print("    Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
