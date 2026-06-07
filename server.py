import json, re, math, os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import nltk
for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(res, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()
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

BOT = FAQChatbot('faqs.json')

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
