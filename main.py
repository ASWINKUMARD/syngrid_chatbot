import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from collections import deque
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import time
import os
import hashlib
from typing import List
from langchain.embeddings.base import Embeddings

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Syngrid AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# STUNNING SYNGRID CUSTOM CSS
# ========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 25%, #2d1b4e 50%, #1a1f3a 75%, #0a0e27 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .syngrid-header {
        background: linear-gradient(135deg, rgba(88, 28, 135, 0.9) 0%, rgba(37, 99, 235, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(147, 51, 234, 0.4), 0 0 100px rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .syngrid-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.03), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .syngrid-logo {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60efff 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 40px rgba(96, 239, 255, 0.5);
    }
    
    .syngrid-tagline {
        color: #c4b5fd;
        font-size: 1.3rem;
        font-weight: 300;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 8px 25px;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
        font-size: 1.1rem;
        animation: slideInRight 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .user-message:hover::before {
        left: 100%;
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        padding: 1.5rem 2rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1.5rem 0;
        color: #1e293b;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border-left: 5px solid #8b5cf6;
        font-size: 1.1rem;
        animation: slideInLeft 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        line-height: 1.7;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px) scale(0.9); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px) scale(0.9); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.9rem 2.8rem;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 40px rgba(139, 92, 246, 0.7);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(1.02);
    }
    
    .stTextInput>div>div>input {
        border-radius: 50px;
        border: 2px solid rgba(139, 92, 246, 0.5);
        padding: 1.2rem 2rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        font-size: 1.1rem;
        color: #e0e7ff;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #a78bfa;
        box-shadow: 0 0 30px rgba(167, 139, 250, 0.4);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #94a3b8;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60efff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
        font-weight: 600;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(167, 139, 250, 0.2);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.15);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.25);
    }
    
    h1, h2, h3 { 
        background: linear-gradient(135deg, #60efff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    p, label { 
        color: #e0e7ff !important;
        line-height: 1.6;
    }
    
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-radius: 25px;
        border: 1px solid rgba(167, 139, 250, 0.2);
        margin: 2rem 0;
    }
    
    .welcome-container h3 {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .welcome-container p {
        font-size: 1.2rem;
        color: #c4b5fd !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #60efff 0%, #a78bfa 50%, #ec4899 100%);
        border-radius: 10px;
    }
    
    .stSpinner > div {
        border-top-color: #a78bfa !important;
    }
    
    ::-webkit-scrollbar { width: 14px; }
    ::-webkit-scrollbar-track { 
        background: rgba(15, 23, 42, 0.5);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(180deg, #8b5cf6 0%, #6366f1 100%);
        border-radius: 10px;
        border: 2px solid rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #a78bfa 0%, #818cf8 100%);
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.5), transparent);
        margin: 2rem 0;
    }
    
    .init-section {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
        padding: 3rem;
        border-radius: 25px;
        border: 1px solid rgba(167, 139, 250, 0.3);
        margin: 2rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# DATABASE SETUP (In-Memory for Cloud)
# ========================================
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ========================================
# CONFIGURATION
# ========================================
# Get API key from environment variable (Streamlit Cloud secrets)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SYNGRID_WEBSITE = "https://syngrid.com/"

# ========================================
# SIMPLE EMBEDDINGS CLASS
# ========================================
class SimpleEmbeddings(Embeddings):
    """Hash-based embeddings - no external models needed"""
    
    def __init__(self):
        self.dimension = 384
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        words = text.lower().split()
        words = [''.join(c for c in word if c.isalnum()) for word in words]
        return [w for w in words if len(w) > 2]
    
    def _get_vector(self, text: str) -> List[float]:
        """Create hash-based embedding vector"""
        tokens = self._tokenize(text)
        vector = [0.0] * self.dimension
        
        for i, token in enumerate(tokens[:25]):
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            positions = [(hash_val + j) % self.dimension for j in range(15)]
            
            for pos in positions:
                vector[pos] += 1.0 / (i + 1)
        
        # Normalize
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self._get_vector(text)

# ========================================
# RAG SERVICE CLASS
# ========================================
class SyngridAI:
    def __init__(self):
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "message": "Not initialized", "pages_scraped": 0}
        self.vectorstore = None

    def scrape_website(self, base_url, max_pages=20, progress_callback=None):
        """Scrape Syngrid website with error handling"""
        visited = set()
        all_content = []
        queue = deque([base_url])
        base_domain = urlparse(base_url).netloc

        while queue and len(visited) < max_pages:
            current_url = queue.popleft()

            if current_url in visited or len(visited) >= max_pages:
                continue

            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                resp = requests.get(current_url, headers=headers, timeout=15, allow_redirects=True)

                if resp.status_code != 200:
                    continue

                visited.add(current_url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages)

                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove unwanted elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 20]
                clean_text = "\n".join(lines)

                if len(clean_text) > 200:
                    all_content.append(f"SOURCE: {current_url}\n\n{clean_text}")

                # Find more links
                if len(visited) < max_pages:
                    for link in soup.find_all("a", href=True):
                        next_url = urljoin(current_url, link["href"])
                        if urlparse(next_url).netloc == base_domain and next_url not in visited:
                            clean_url = next_url.split('#')[0].split('?')[0]
                            if clean_url not in visited and clean_url not in queue:
                                queue.append(clean_url)

            except Exception as e:
                continue

        self.status["pages_scraped"] = len(visited)
        return "\n\n=== PAGE BREAK ===\n\n".join(all_content)

    def initialize(self, url, max_pages, progress_callback=None):
        """Initialize Syngrid AI Assistant"""
        try:
            # Check API key
            if not OPENROUTER_API_KEY:
                self.status["message"] = "API key not configured"
                return False
            
            self.status["message"] = "🔍 Analyzing Syngrid website..."
            content = self.scrape_website(url, max_pages, progress_callback)

            if len(content) < 500:
                self.status["message"] = "❌ Insufficient content scraped"
                return False

            self.status["message"] = "📝 Processing information..."
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150,
                separators=["\n\n=== PAGE BREAK ===\n\n", "\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_text(content)

            if len(chunks) < 5:
                self.status["message"] = "❌ Not enough content chunks"
                return False

            self.status["message"] = "🧠 Building knowledge base..."
            
            embeddings = SimpleEmbeddings()

            # Create vectorstore without persistence
            self.vectorstore = Chroma.from_texts(
                chunks,
                embedding=embeddings
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            self.status["ready"] = True
            self.status["message"] = "✅ Assistant Ready!"
            return True

        except Exception as e:
            self.status["message"] = f"❌ Error: {str(e)}"
            return False

    def ask(self, question):
        """Ask Syngrid AI Assistant"""
        if not self.status["ready"]:
            return "⚡ Please initialize the assistant first by clicking the Initialize button above."

        if not OPENROUTER_API_KEY:
            return "⚠️ API key not configured. Please set OPENROUTER_API_KEY in Streamlit secrets."

        q_lower = question.lower().strip()
        
        # Check cache
        if q_lower in self.cache:
            return self.cache[q_lower]

        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)
            
            if not docs:
                return "I couldn't find relevant information about that. Could you rephrase your question or ask about Syngrid's services, technologies, or solutions?"

            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            prompt = f"""You are Syngrid AI Assistant, a helpful expert on Syngrid Technologies. Answer based on the context provided.

Context:
{context[:3500]}

Question: {question}

Provide a clear, helpful answer (2-4 sentences). Focus on being accurate and informative:"""

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "google/gemini-flash-1.5-8b",
                "messages": [
                    {"role": "system", "content": "You are Syngrid AI Assistant. Provide helpful, accurate answers about Syngrid Technologies."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 250
            }

            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if res.status_code == 200:
                answer = res.json()["choices"][0]["message"]["content"].strip()
                self.cache[q_lower] = answer
                
                # Store in database
                try:
                    db = SessionLocal()
                    chat_entry = ChatHistory(
                        question=question,
                        answer=answer,
                        timestamp=datetime.now(timezone.utc)
                    )
                    db.add(chat_entry)
                    db.commit()
                    db.close()
                except:
                    pass
                
                return answer
            else:
                return "⚠️ I'm having trouble processing that right now. Please try again in a moment."

        except Exception as e:
            return f"⚠️ An error occurred: {str(e)[:100]}"

# ========================================
# SESSION STATE
# ========================================
if 'ai' not in st.session_state:
    st.session_state.ai = SyngridAI()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# ========================================
# HEADER
# ========================================
st.markdown("""
<div class="syngrid-header">
    <div class="syngrid-logo">⚡ SYNGRID AI ASSISTANT</div>
    <div class="syngrid-tagline">Your Intelligent Guide to Syngrid Technologies</div>
</div>
""", unsafe_allow_html=True)

# ========================================
# INITIALIZATION SECTION
# ========================================
if not st.session_state.initialized:
    st.markdown('<div class="init-section">', unsafe_allow_html=True)
    st.markdown("### 🚀 Welcome to Syngrid AI Assistant")
    st.markdown("Initialize the AI assistant to start learning about Syngrid Technologies' innovative solutions, services, and expertise.")
    
    if not OPENROUTER_API_KEY:
        st.error("⚠️ **API Key Required**: Please configure OPENROUTER_API_KEY in Streamlit Cloud secrets to use this application.")
        st.info("📖 **Setup Instructions**: Go to Settings → Secrets in your Streamlit Cloud dashboard and add: `OPENROUTER_API_KEY = \"your-key-here\"`")
    
    st.markdown("")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("⚡ Initialize Assistant", use_container_width=True, key="init_btn", disabled=not OPENROUTER_API_KEY):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.markdown(
                    f"<p style='text-align:center; color:#c4b5fd; font-size:1.1rem'>⚡ Analyzing: {current}/{total} pages</p>",
                    unsafe_allow_html=True
                )
            
            with st.spinner("🔮 Initializing Syngrid AI..."):
                success = st.session_state.ai.initialize(
                    SYNGRID_WEBSITE,
                    20,
                    update_progress
                )
                
                if success:
                    st.session_state.initialized = True
                    st.success("✅ Assistant Ready! Reloading...")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"❌ {st.session_state.ai.status['message']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================
# CHAT INTERFACE
# ========================================
else:
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ Status", "Online", delta="Ready")
    with col2:
        st.metric("📄 Pages Indexed", st.session_state.ai.status["pages_scraped"])
    with col3:
        st.metric("💬 Conversations", len(st.session_state.messages))
    
    st.markdown("---")
    
    # Chat Container
    chat_container = st.container()
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div class="welcome-container">
                <h3>👋 Welcome to Syngrid AI!</h3>
                <p>
                    Ask me anything about Syngrid Technologies:<br>
                    Our services • Solutions • Technologies • Expertise • Projects
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for q, a in st.session_state.messages:
                st.markdown(f'<div class="user-message">👤 {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">⚡ {a}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input Section
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            key="user_input",
            placeholder="💭 Ask about Syngrid's services, technologies, solutions...",
            label_visibility="collapsed"
        )
    
    with col2:
        send = st.button("⚡ Send", use_container_width=True)
    
    if send and user_input and user_input.strip():
        with st.spinner("⚡ Thinking..."):
            response = st.session_state.ai.ask(user_input.strip())
            st.session_state.messages.append((user_input.strip(), response))
            st.rerun()
    
    # Action Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col3:
        if st.button("⚙️ Restart Assistant", use_container_width=True):
            st.session_state.messages = []
            st.session_state.initialized = False
            st.session_state.ai = SyngridAI()
            st.rerun()

# ========================================
# FOOTER
# ========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 2rem;'>
    ⚡ Powered by Syngrid Technologies | AI-Driven Innovation
</div>
""", unsafe_allow_html=True)
