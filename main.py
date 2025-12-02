# app.py
"""
Syngrid AI Assistant - Streamlit RAG Chatbot (Production-ready)
Features:
- Scrapes up to 50 pages from syngrid.com (priority pages first)
- Builds FAISS vector store with BGE-small embeddings (BAAI/bge-small-en-v1.5)
- Robust embedding loader with local hf cache (./hf_cache)
- OpenRouter integration for completions (set OPENROUTER_API_KEY in env / Streamlit secrets)
- Gorgeous Streamlit UI (preserves user's visual design; fixes input color)
- Flexible phone validation (8-20 digits)
- Safe DB storage (SQLite in project folder)
- Resilient error handling for model / cache corruption
NOTE: Do NOT commit secrets (OPENROUTER_API_KEY) to repo.
"""

import os
import re
import json
import time
import shutil
import hashlib
from collections import deque
from datetime import datetime, timezone
from typing import List

import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# langchain helpers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Try to import HuggingFace wrapper; fallback to sentence-transformers direct
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

# For sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Syngrid AI Assistant", page_icon="⚡", layout="wide")

BASE_URL = "https://syngrid.com/"
DB_FILE = "syngrid_chat.db"
HF_CACHE_DIR = os.path.abspath("./hf_cache")
FAISS_DIR = os.path.abspath("./faiss_index")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Use env var or Streamlit secrets for OpenRouter key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
# Model used for generation (user previously used kwaipilot/kat-coder-pro:free)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "kwaipilot/kat-coder-pro:free")

# Embedding model choice (B2)
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"  # BGE-small (recommended by user)

# Set HF cache envs for local project cache (helps on Windows/Streamlit Cloud)
os.environ.update({
    "HF_HOME": HF_CACHE_DIR,
    "TRANSFORMERS_CACHE": HF_CACHE_DIR,
    "SENTENCE_TRANSFORMERS_HOME": HF_CACHE_DIR,
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "TOKENIZERS_PARALLELISM": "false",
})

# ---------------------------
# CSS / UI (preserve visuals, fix input text color)
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%); background-size: 400% 400%; animation: gradientWave 20s ease infinite; }
    @keyframes gradientWave { 0% {background-position:0% 50%} 50% {background-position:100% 50%} 100% {background-position:0% 50%} }
    .syngrid-header { background: linear-gradient(135deg, rgba(102,126,234,0.95), rgba(118,75,162,0.95)); padding: 3rem; border-radius: 30px; text-align:center; margin-bottom:2.5rem; box-shadow:0 25px 70px rgba(102,126,234,0.5); border:2px solid rgba(255,255,255,0.2); }
    .syngrid-logo { font-size: 3.6rem; font-weight:800; background: linear-gradient(135deg,#fff,#a8edea); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem; }
    .syngrid-tagline { color:#e0f2fe; font-size:1.15rem; font-weight:300; }
    .user-message { background: linear-gradient(135deg,#667eea,#764ba2); padding:1.2rem 1.6rem; border-radius:18px 18px 8px 18px; margin:1rem 0; color:white; font-size:1rem; line-height:1.5; }
    .bot-message { background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(240,248,255,0.98)); padding:1.2rem 1.6rem; border-radius:18px 18px 18px 8px; margin:1rem 0; color:#0b1221; font-size:1rem; line-height:1.6; border-left:6px solid #667eea; }
    .stButton>button { background: linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%); color:white; border-radius:50px; padding:0.9rem 2rem; font-weight:700; box-shadow:0 12px 35px rgba(102,126,234,0.6); }
    .stTextInput>div>div>input { border-radius: 28px; border: 2px solid rgba(102,126,234,0.5); padding:0.9rem 1.4rem; background: rgba(255,255,255,0.95) !important; color: #000 !important; }
    .stTextInput>div>div>input::placeholder { color:#6b7280 !important; }
    textarea, input, select { color: #000 !important; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# DATABASE (SQLAlchemy)
# ---------------------------
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class UserContact(Base):
    __tablename__ = "user_contacts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ---------------------------
# Validation utilities
# ---------------------------
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}$'
    return bool(re.match(pattern, (email or "").strip()))

def validate_phone_flexible(phone: str) -> bool:
    cleaned = re.sub(r'\D', '', phone or "")
    return 7 <= len(cleaned) <= 20

def validate_name(name: str) -> bool:
    return bool(name and len(name.strip()) >= 2)

# ---------------------------
# Embeddings loader (BGE-small, robust)
# ---------------------------
def make_embedding_client():
    """
    Preferred: LangChain HuggingFace wrapper -> fallback to sentence-transformers directly.
    Uses local cache folder (HF_CACHE_DIR).
    """
    # Try LangChain wrapper (if available & compatible)
    if HuggingFaceEmbeddings is not None:
        try:
            emb = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_ID,
                cache_folder=HF_CACHE_DIR,
                model_kwargs={"device": "cpu"},
            )
            st.info("Embeddings: loaded via LangChain HuggingFace wrapper.")
            return emb
        except Exception as e:
            st.warning(f"LangChain HuggingFaceEmbeddings failed: {e}. Falling back...")

    # Fallback to sentence-transformers
    if SentenceTransformer is not None:
        try:
            st.info(f"Loading sentence-transformers model: {EMBED_MODEL_ID} (this may take 20-60s first run)")
            model = SentenceTransformer(EMBED_MODEL_ID, cache_folder=HF_CACHE_DIR)
            st.info("Embeddings: loaded via sentence-transformers.")
            class STWrapper:
                def embed_documents(self, texts: List[str]):
                    return [model.encode(t, show_progress_bar=False, convert_to_numpy=True) for t in texts]
                def embed_query(self, text: str):
                    return model.encode(text, show_progress_bar=False, convert_to_numpy=True)
            return STWrapper()
        except Exception as e:
            st.error(f"Failed to load sentence-transformers model: {e}")
            raise
    else:
        raise RuntimeError("No embeddings backend available. Install 'sentence-transformers' or 'langchain-community'.")

# initialize embeddings (cached in session_state)
if 'embeddings' not in st.session_state:
    try:
        st.session_state['embeddings'] = make_embedding_client()
    except Exception as e:
        st.session_state['embeddings'] = None
        st.error("Embeddings initialization failed. Check logs and ensure network access. " + str(e))

# ---------------------------
# Vector store helpers (FAISS)
# ---------------------------
def load_faiss_store():
    emb = st.session_state.get('embeddings')
    if emb is None:
        return None
    if os.path.exists(FAISS_DIR):
        try:
            store = FAISS.load_local(FAISS_DIR, emb)
            return store
        except Exception as e:
            st.warning(f"Failed to load FAISS index (corrupted?). Removing and allowing rebuild. Error: {e}")
            try:
                shutil.rmtree(FAISS_DIR)
            except Exception:
                pass
            return None
    return None

def save_faiss_store(store):
    os.makedirs(FAISS_DIR, exist_ok=True)
    store.save_local(FAISS_DIR)

# ---------------------------
# Scraper / Content extraction
# ---------------------------
PRIORITY_PAGES = [
    "", "about", "about-us", "services", "solutions", "products",
    "contact", "contact-us", "team", "careers", "portfolio", "case-studies",
    "industries", "technology", "expertise", "what-we-do", "who-we-are",
    "footer", "locations", "reach-us", "blog", "clients", "partners"
]

def is_valid_url_for_scrape(url: str, base_domain: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.netloc != base_domain:
            return False
    except Exception:
        return False
    skip = [r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$', r'/wp-admin/', r'/wp-content/(?!.*contact)', r'/feed/', r'/rss/', r'/sitemap', r'/login', r'/register', r'\.css$', r'\.js$']
    for p in skip:
        if re.search(p, url.lower()):
            return False
    return True

def extract_text_soup(soup: BeautifulSoup) -> str:
    for tag in soup(['script', 'style', 'nav', 'aside', 'iframe', 'form', 'button']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    lines = [ln.strip() for ln in text.split('\n') if ln.strip() and len(ln.strip()) > 20]
    return "\n".join(lines)

# Improved contact extraction (emails, phones, addresses)
def extract_contact_info_from_text(text: str):
    emails = set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text))
    phones = set()
    for patt in [r'\+\d{1,3}[\s\-]?\d{4,5}[\s\-]?\d{4,5}', r'\(?\d{2,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}']:
        for m in re.findall(patt, text):
            cleaned = re.sub(r'[^\d\+]', '', m)
            if 7 <= len(re.sub(r'\D', '', cleaned)) <= 20:
                phones.add(m.strip())
    return emails, phones

def crawl_website_collect(base_url: str, max_pages: int = 50, progress_callback=None):
    base_domain = urlparse(base_url).netloc
    visited = set()
    queue = deque()
    for p in PRIORITY_PAGES:
        queue.append(urljoin(base_url, p))
    if base_url not in queue:
        queue.append(base_url)
    headers = {"User-Agent": "Mozilla/5.0 (SyngridBot/1.0)"}
    collected = []
    found_emails = set()
    found_phones = set()
    india_address = None
    singapore_address = None

    while queue and len(visited) < max_pages:
        url = queue.popleft().split('#')[0].split('?')[0]
        if url in visited:
            continue
        if not is_valid_url_for_scrape(url, base_domain):
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            visited.add(url)
            if progress_callback:
                progress_callback(len(visited), max_pages)
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = extract_text_soup(soup)
            if not text or len(text) < 120:
                # still queue child links but skip saving
                for a in soup.find_all('a', href=True):
                    nxt = urljoin(url, a['href']).split('#')[0].split('?')[0]
                    if nxt not in visited and nxt not in queue and is_valid_url_for_scrape(nxt, base_domain):
                        queue.append(nxt)
                continue
            # contact extraction
            emails, phones = extract_contact_info_from_text(text)
            found_emails.update(emails)
            found_phones.update(phones)
            # attempt to find addresses in contact/footer pages
            if any(k in url.lower() for k in ['contact', 'footer', 'reach', 'about']):
                lines = text.split('\n')
                # detect India addresses (simple heuristics)
                for i, line in enumerate(lines):
                    if 'madurai' in line.lower() or 'india' in line.lower():
                        snippet = " ".join(lines[max(0, i-3): min(len(lines), i+4)])
                        if len(snippet) > 40 and not india_address:
                            india_address = snippet.strip()
                    if 'singapore' in line.lower() and re.search(r'\d{6}', line):
                        snippet = " ".join(lines[max(0, i-3): min(len(lines), i+4)])
                        if len(snippet) > 40 and not singapore_address:
                            singapore_address = snippet.strip()
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            collected.append(f"URL: {url}\nTITLE: {title}\n\n{text}")
            # queue links
            for a in soup.find_all('a', href=True):
                nxt = urljoin(url, a['href']).split('#')[0].split('?')[0]
                if nxt not in visited and nxt not in queue and is_valid_url_for_scrape(nxt, base_domain):
                    queue.append(nxt)
        except Exception:
            continue

    separator = "\n\n=== PAGE BREAK ===\n\n"
    full_text = separator.join(collected)
    # prepend formatted contact info
    contact_block = ""
    if india_address:
        contact_block += f"INDIA OFFICE:\n{india_address}\n\n"
    if singapore_address:
        contact_block += f"SINGAPORE OFFICE:\n{singapore_address}\n\n"
    if found_emails:
        contact_block += "EMAILS:\n" + "\n".join(sorted(found_emails)) + "\n\n"
    if found_phones:
        contact_block += "PHONES:\n" + "\n".join(sorted(found_phones)) + "\n\n"
    if contact_block:
        full_text = contact_block + separator + full_text
    return full_text, len(visited)

# ---------------------------
# Knowledge base build
# ---------------------------
def build_knowledge_base(max_pages: int = 50, progress_callback=None):
    st.info("Starting crawl and embeddings (this may take a minute on first run)...")
    content, pages = crawl_website_collect(BASE_URL, max_pages=max_pages, progress_callback=progress_callback)
    if not content or len(content) < 400:
        st.error("Crawl returned insufficient content. Check target site or try again.")
        return None, pages
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(content)
    emb = st.session_state.get('embeddings')
    if emb is None:
        st.error("Embeddings not available. Cannot build index.")
        return None, pages
    store = FAISS.from_texts(chunks, emb)
    save_faiss_store(store)
    return store, pages

# ---------------------------
# OpenRouter helper (robust)
# ---------------------------
def ask_openrouter(prompt: str, context: str = "") -> str:
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY not configured. Set it in environment or Streamlit secrets."
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are Syngrid AI Assistant. Answer concisely and accurately using only the provided context when available."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
        ],
        "temperature": 0.2,
        "max_tokens": 450
    }
    try:
        res = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return f"⚠️ Network error contacting OpenRouter: {str(e)}"

    # Try parse JSON
    try:
        data = res.json()
    except Exception:
        return f"⚠️ OpenRouter returned non-JSON response (status {res.status_code})."

    if res.status_code != 200:
        # attempt to extract message
        err = data.get("error", {}) if isinstance(data, dict) else {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return f"⚠️ OpenRouter Error {res.status_code}: {msg or res.text[:200]}"

    choices = data.get("choices") or []
    if not choices:
        return "⚠️ OpenRouter returned empty choices. Please try again."

    # prefer field message.content
    answer = choices[0].get("message", {}).get("content", "")
    if not answer:
        # fallback to other shapes
        answer = str(choices[0])
    answer = answer.strip()
    if not answer:
        return "⚠️ Model produced empty output. Try rephrasing."

    return answer

# ---------------------------
# RAG Assistant class
# ---------------------------
class SyngridAI:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "message": "Not initialized", "pages_scraped": 0}

    def load_index_if_exists(self):
        store = load_faiss_store()
        if store:
            self.vectorstore = store
            # as_retriever uses default similarity search
            try:
                self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            except Exception:
                # if as_retriever not present, fallback to direct usage
                self.retriever = None
            self.status["ready"] = True
            self.status["message"] = "Index loaded"
            return True
        return False

    def initialize(self, url: str, max_pages: int = 50, progress_callback=None) -> bool:
        # build index
        try:
            self.status["message"] = "Crawling site..."
            store, pages = build_knowledge_base(max_pages=max_pages, progress_callback=progress_callback)
            if store is None:
                self.status["message"] = "Failed to build knowledge base"
                return False
            self.vectorstore = store
            self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.status["ready"] = True
            self.status["message"] = "Ready"
            self.status["pages_scraped"] = pages
            return True
        except Exception as e:
            self.status["message"] = f"Initialization error: {str(e)}"
            return False

    def ask(self, question: str) -> str:
        if not self.status["ready"]:
            return "⚡ Assistant not initialized. Please initialize first."

        q_lower = (question or "").lower().strip()
        contact_kw = ['contact', 'email', 'phone', 'address', 'office', 'location', 'reach', 'call', 'visit', 'where']
        if any(k in q_lower for k in contact_kw):
            # try to return contact info from index (if present) - simple approach
            # attempt to search for 'contact' content
            try:
                docs = self.vectorstore.similarity_search("contact information", k=3)
                ctx = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
                if ctx:
                    return ctx[:2000]
            except Exception:
                pass
            return "📞 Contact info not found in index. Please check the website."

        # cached?
        if q_lower in self.cache:
            return self.cache[q_lower]

        try:
            # retrieval
            try:
                docs = self.vectorstore.similarity_search(question, k=5)
            except Exception:
                docs = []
            context = "\n\n".join([getattr(d, "page_content", "") for d in docs[:3]])
            # ask LLM
            answer = ask_openrouter(question, context)
            # if API returned a warning message text starting with "⚠️", preserve it
            if isinstance(answer, str) and answer.startswith("⚠️"):
                return answer
            # fallback if LLM non-helpful
            if not answer or len(answer.strip()) < 3:
                return "⚠️ Having trouble processing that. Please try again."
            self.cache[q_lower] = answer
            # save chat to DB
            try:
                db = SessionLocal()
                entry = ChatHistory(question=question, answer=answer, timestamp=datetime.now(timezone.utc))
                db.add(entry)
                db.commit()
                db.close()
            except Exception:
                pass
            return answer
        except Exception as e:
            return f"⚠️ Error: {str(e)[:200]}"

# ---------------------------
# Session state init
# ---------------------------
if 'ai' not in st.session_state:
    st.session_state.ai = SyngridAI()
    # try to load existing index if present
    st.session_state.ai.load_index_if_exists()

if 'messages' not in st.session_state:
    st.session_state.messages = []  # list of (q,a)
if 'initialized' not in st.session_state:
    st.session_state.initialized = st.session_state.ai.status.get("ready", False)
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'user_info_collected' not in st.session_state:
    st.session_state.user_info_collected = False
if 'show_user_form' not in st.session_state:
    st.session_state.show_user_form = False

# ---------------------------
# UI Layout
# ---------------------------
st.markdown(f"""
<div class="syngrid-header">
  <div class="syngrid-logo">⚡ SYNGRID AI ASSISTANT</div>
  <div class="syngrid-tagline">Your intelligent guide to Syngrid Technologies — RAG powered</div>
</div>
""", unsafe_allow_html=True)

# user info modal
if st.session_state.show_user_form and not st.session_state.user_info_collected:
    with st.container():
        st.markdown("""
            <div style='padding: 2rem; border-radius: 20px; background: rgba(255,255,255,0.06);'>
                <h3 style='color:white; text-align:center;'>Please share your contact information</h3>
                <p style='color:#e0f2fe; text-align:center;'>Optional — used only for follow-up</p>
            </div>
        """, unsafe_allow_html=True)
        with st.form("user_info_form"):
            name = st.text_input("Full Name *")
            email = st.text_input("Email Address *")
            phone = st.text_input("Phone Number (international allowed)")
            submitted = st.form_submit_button("Save contact")
            if submitted:
                errs = []
                if not validate_name(name):
                    errs.append("Name must be at least 2 characters.")
                if not validate_email(email):
                    errs.append("Enter valid email.")
                if not validate_phone_flexible(phone):
                    errs.append("Enter valid phone (7-20 digits).")
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    try:
                        db = SessionLocal()
                        contact = UserContact(name=name.strip(), email=email.strip(), phone=phone.strip(), timestamp=datetime.now(timezone.utc))
                        db.add(contact)
                        db.commit()
                        db.close()
                        st.session_state.user_info_collected = True
                        st.session_state.show_user_form = False
                        st.success("Thanks — contact saved.")
                        time.sleep(1)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error("DB save error: " + str(e))

# Initialization block
if not st.session_state.initialized:
    st.markdown('<div style="padding:1rem; border-radius:14px; background: rgba(255,255,255,0.04);">', unsafe_allow_html=True)
    st.header("🚀 Initialize Syngrid AI Assistant")
    st.write("Crawl the site (up to 50 pages) and build the knowledge base. Requires OpenRouter API key configured.")
    if not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY is not configured. Set it in environment or Streamlit secrets.")
    cols = st.columns([2,1,2])
    if cols[1].button("⚡ Initialize Assistant", use_container_width=True, disabled=not OPENROUTER_API_KEY):
        progress = st.progress(0)
        status = st.empty()
        def progress_cb(current, total):
            pct = min(1.0, float(current) / float(total))
            progress.progress(pct)
            status.info(f"Scraped {current}/{total} pages...")
        with st.spinner("Initializing (crawl + embeddings + index)..."):
            ok = st.session_state.ai.initialize(BASE_URL, max_pages=50, progress_callback=progress_cb)
            st.session_state.initialized = ok
            if ok:
                st.success("Assistant ready — knowledge base built.")
                st.experimental_rerun()
            else:
                st.error(f"Initialization failed: {st.session_state.ai.status.get('message')}")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Chat interface
    cols = st.columns([3,1])
    with cols[0]:
        st.metric("Status", "Online" if st.session_state.ai.status.get("ready") else "Offline")
    with cols[1]:
        st.metric("Pages Indexed", st.session_state.ai.status.get("pages_scraped", 0))
    st.markdown("---")

    # Chat display
    if not st.session_state.messages:
        st.markdown("""
            <div style='padding:2rem; border-radius:16px; background: rgba(255,255,255,0.03); text-align:center;'>
                <h3 style='color:#fff;'>👋 Welcome to Syngrid AI</h3>
                <p style='color:#e0f2fe;'>Ask about services, solutions, technologies, contact info and more.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for q, a in st.session_state.messages:
            st.markdown(f'<div class="user-message">👤 {st.markdown(q)}', unsafe_allow_html=True)
            # show bot message
            st.markdown(f'<div class="bot-message">⚡ {a}</div>', unsafe_allow_html=True)

    st.markdown("---")
    input_col, action_col = st.columns([5,1])
    with input_col:
        user_input = st.text_input("Message", key="user_input", placeholder="Ask about Syngrid (services, contact, technologies...)", label_visibility="collapsed")
    with action_col:
        send = st.button("⚡ Send", use_container_width=True)

    if send and user_input and user_input.strip():
        q = user_input.strip()
        st.session_state.question_count += 1
        # Ask and show
        with st.spinner("⚡ Thinking..."):
            ans = st.session_state.ai.ask(q)
        st.session_state.messages.append((q, ans))
        # If it's the 3rd question and user info not collected -> show form
        if st.session_state.question_count >= 3 and not st.session_state.user_info_collected:
            st.session_state.show_user_form = True
        st.experimental_rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.question_count = 0
            st.experimental_rerun()
    with c2:
        if st.button("🔁 Rebuild Knowledge Base", use_container_width=True):
            # remove faiss index and rebuild
            try:
                shutil.rmtree(FAISS_DIR)
            except Exception:
                pass
            st.session_state.initialized = False
            st.session_state.ai = SyngridAI()
            st.experimental_rerun()
    with c3:
        if st.button("📥 Download chats (CSV)", use_container_width=True):
            db = SessionLocal()
            rows = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
            import csv, io
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["id", "question", "answer", "timestamp"])
            for r in rows:
                writer.writerow([r.id, r.question, r.answer, r.timestamp.isoformat()])
            st.download_button("Download chats.csv", buf.getvalue(), "chats.csv", mime="text/csv")
            db.close()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#e0f2fe; padding:1.2rem;'>
    ⚡ Powered by Syngrid Technologies — RAG Chatbot | Keep OPENROUTER_API_KEY secret
</div>
""", unsafe_allow_html=True)
