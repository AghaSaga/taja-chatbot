from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re, math, docx, traceback, requests
from bs4 import BeautifulSoup
from collections import Counter
from dotenv import load_dotenv
from google import genai

# ─────────────── Config ───────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

# Comma-separated allow-list of pages you want included (optional)
# Example .env:
#   TAJA_WEBSITES=https://manataja.us,https://manataja.us/membership/
ALLOWED_SITES = [u.strip() for u in (os.getenv("TAJA_WEBSITES") or "").split(",") if u.strip()]

client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)  # frontend runs at 8000, backend at 5001

DOC_PATH = "taja_data/Manataja.docx"

# ─────────────── Loaders ───────────────
def load_docx_text(path: str) -> str:
    """Read paragraphs + tables from .docx."""
    doc = docx.Document(path)
    parts = []

    # paragraphs
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    # tables (each row -> one line)
    for table in doc.tables:
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join([c for c in cells if c]))

    return "\n".join(parts)

def fetch_site_text(url: str, timeout: int = 12) -> str:
    """Fetch & extract visible text from an allowed page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (TAJA-Chatbot)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)
    except Exception as e:
        print(f"[WEB] Skip {url}: {e}")
        return ""

# ─────────────── Build corpus (DOCX + optional WEB) ───────────────
def build_corpus():
    global CORPUS_TEXT, CHUNKS, CHUNK_TOKS, CHUNK_VECS, CHUNK_NORMS

    raw_doc = load_docx_text(DOC_PATH)
    raw_web = ""
    if ALLOWED_SITES:
        fetched = []
        for u in ALLOWED_SITES:
            txt = fetch_site_text(u)
            if txt:
                fetched.append(f"[PAGE] {u}\n{txt}")
        raw_web = "\n\n".join(fetched)

    CORPUS_TEXT = "\n\n".join([x for x in [raw_doc, raw_web] if x])

    # index
    CHUNKS = chunk_text(CORPUS_TEXT, 1000)
    CHUNK_TOKS = [normalize(c).split() for c in CHUNKS]
    CHUNK_VECS = [Counter(t) for t in CHUNK_TOKS]
    CHUNK_NORMS = [l2(v) for v in CHUNK_VECS]

    print(f"[TAJA] Corpus chars: {len(CORPUS_TEXT):,}  | chunks: {len(CHUNKS)}")
    print("[TAJA] contains 'membership'?", "membership" in CORPUS_TEXT.lower(),
          "| contains 'fee'/'fees'?", ("fee" in CORPUS_TEXT.lower() or "fees" in CORPUS_TEXT.lower()))

# ─────────────── Retrieval helpers ───────────────
def normalize(t):
    return re.sub(r"[^a-z0-9\s]", " ", (t or "").lower())

def chunk_text(txt, target_chars=1000):
    paras = [p for p in (txt or "").split("\n") if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur_len + len(p) > target_chars and cur:
            chunks.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += len(p)
    if cur: chunks.append("\n".join(cur))
    return chunks

def l2(c):
    return math.sqrt(sum(v*v for v in c.values())) or 1e-9

STOP = set(("the a an and or for of to in on with by from at as about into over after before within without "
            "between among is are was were be been being it this that these those you your their our we they "
            "he she them us").split())

def score_chunks(q, k=5):
    toks = [t for t in normalize(q).split() if t not in STOP]
    if not toks:
        return [(0.0, i) for i in range(min(k, len(CHUNKS)))]
    qv = Counter(toks); qn = l2(qv)
    scores = []
    for i, cv in enumerate(CHUNK_VECS):
        dot = sum(qv[t]*cv.get(t,0) for t in qv)
        cos = dot / (qn * CHUNK_NORMS[i])
        overlap = sum(1 for t in qv if cv.get(t,0)>0)
        scores.append((cos + 0.02*overlap, i))
    scores.sort(reverse=True)
    return scores[:k]

def build_context(q, k=5, cap=5500):
    picks = score_chunks(q, k)
    chosen = [CHUNKS[i] for s,i in picks if s>0] or [CHUNKS[i] for s,i in picks]
    ctx = ""
    for ch in chosen:
        if len(ctx)+len(ch)+100 < cap: ctx += "\n\n---\n"+ch
        else: break
    return ctx.strip() or CORPUS_TEXT[:cap]

# ─────────────── Fees-fallback helpers ───────────────
def wants_fees(q: str) -> bool:
    ql = (q or "").lower()
    return bool(re.search(r'\b(fee|fees|dues|price|cost)\b', ql))

def context_mentions_fees(ctx: str) -> bool:
    cl = (ctx or "").lower()
    return ("fee" in cl or "fees" in cl or "dues" in cl or "$" in cl or "usd" in cl or "dollar" in cl)

# ─────────────── Prompting ───────────────
SYSTEM = (
    "You are TAJA's official AI assistant. "
    "Use ONLY the provided TAJA sources (document + approved website pages). "
    "If a detail is missing or ambiguous, say: "
    "\"I couldn't find that in the official TAJA sources. Please contact TAJA for confirmation.\" "
    "Answer first in one short sentence, then provide concise bullets. "
    "Prefer short headings; avoid repeating the same list multiple times."
)

FORMAT = ("Format with: brief answer-first line; clear headings (Membership, Events, Contacts, Fees); "
          "bulleted lists; simple tables only if content is tabular; keep links as full URLs.")

def make_prompt(q, ctx):
    return (f"{SYSTEM}\n\n{FORMAT}\n\n=== OFFICIAL TAJA CONTENT (retrieved) ===\n{ctx}\n=== END CONTENT ===\n\n"
            f"User question: {q}\n\nTAJA Bot:")

# ─────────────── Routes ───────────────
@app.route("/health")
def health():
    return "ok", 200

@app.route("/reload", methods=["POST"])
def reload_corpus():
    try:
        build_corpus()
        return jsonify({"status": "reloaded"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("message") or "").strip()
        if not q:
            return jsonify({"response": "❌ No input provided."}), 400

        # Retrieve a bit wider to improve recall
        ctx = build_context(q, k=8)

        # If user asks about fees but no fee-like tokens in context,
        # pivot to a friendly membership summary WITHOUT saying anything is missing.
        prompt_q = q
        if wants_fees(q) and not context_mentions_fees(ctx):
            extra_ctx = build_context(
                "how to become a member membership registration link membership types contacts email",
                k=6
            )
            ctx = (ctx + "\n\n---\n" + extra_ctx).strip()
            prompt_q = (
                q +
                " If specific dollar amounts are not present in the official content, "
                "DO NOT mention that fees are missing. "
                "Instead, respond with a short heading like 'Membership information' "
                "and provide: how to become a member, the official registration link, "
                "membership types, and relevant contact emails—all strictly from the provided content."
            )

        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=make_prompt(prompt_q, ctx)
        )
        text = (res.text or "").strip() or \
               "Please try rephrasing your question."

        return jsonify({"response": text}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"⚠️ Server error: {e}"}), 500

@app.route("/index_with_chatbot.html", methods=["GET","POST"])
def index_alias():
    return jsonify({"message":"This is the backend. Open http://127.0.0.1:8000/index_with_chatbot.html"}), 200

# ─────────────── Boot ───────────────
if __name__ == "__main__":
    build_corpus()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
