# CRITICAL FIX: Must be FIRST imports before anything else
# This fixes SQLite version issue on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
import re
import os
import time

# Database Configuration
DATABASE_URL = "sqlite:///./syngrid_chat.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

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
    phone = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
SYNGRID_WEBSITE = "https://syngrid.com/"
MODEL = "kwaipilot/kat-coder-pro:free"

PRIORITY_PAGES = [
    "", "about", "services", "solutions", "products", "contact", "team",
    "careers", "blog", "case-studies", "portfolio", "industries",
    "technology", "expertise", "what-we-do", "who-we-are", "footer"
]

class SyngridAI:
    def __init__(self):
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "pages_scraped": 0}
        self.scraped_content = {}
        self.company_info = {
            'emails': set(), 'phones': set(),
            'india_address': None, 'singapore_address': None
        }

    def clean_address(self, text):
        text = ' '.join(text.split())
        text = re.sub(r'(Corporate Office|Branch Office|Head Office|Registered Office)', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_contact_info(self, soup, text, url):
        # Emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif')):
                self.company_info['emails'].add(email.lower())

        # Phones
        phone_patterns = [
            r'\+91\s*\d{5}\s*\d{5}',
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if 10 <= len(cleaned) <= 15:
                    self.company_info['phones'].add(phone.strip())

        # Detect addresses
        lines = text.split('\n')
        for i, line in enumerate(lines):
            low = line.lower()
            if 'madurai' in low and '625' in line and not self.company_info['india_address']:
                block = " ".join(lines[max(0, i-2):min(len(lines), i+5)])
                cleaned = self.clean_address(block)
                if 20 < len(cleaned) < 300:
                    self.company_info['india_address'] = cleaned

            if 'singapore' in low and re.search(r'\d{6}', line) and not self.company_info['singapore_address']:
                block = " ".join(lines[max(0, i-2):min(len(lines), i+5)])
                cleaned = self.clean_address(block)
                if 20 < len(cleaned) < 300:
                    self.company_info['singapore_address'] = cleaned
    
    def is_valid_url(self, url, base_domain):
        try:
            parsed = urlparse(url)
            if parsed.netloc != base_domain:
                return False
            skip = [
                r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$',
                r'/wp-admin/', r'/wp-includes/', r'/login', r'/register',
                r'/cart/', r'/checkout/', r'/feed/', r'/rss/'
            ]
            return not any(re.search(pattern, url.lower()) for pattern in skip)
        except:
            return False

    def extract_content(self, soup, url):
        content_dict = {'url': url, 'title': '', 'main_content': '', 'metadata': {}}

        try:
            t = soup.find('title')
            if t: 
                content_dict['title'] = t.get_text(strip=True)

            meta = soup.find('meta', attrs={"name": "description"})
            if meta and meta.get("content"):
                content_dict['metadata']['description'] = meta["content"]

            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(soup, full_text, url)

            for tag in soup(['script', 'style', 'nav', 'aside', 'iframe', 'noscript', 'form']):
                tag.decompose()

            content_selectors = [
                "main", "article", "[role='main']", ".content",
                ".main-content", "#content", "#main"
            ]
            parts = []
            for sel in content_selectors:
                parts.extend(soup.select(sel))

            main_content = soup.find("body") if not parts else soup.new_tag("div")
            if parts:
                for p in parts:
                    main_content.append(p)

            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
                content_dict['main_content'] = "\n".join(lines)
        except Exception as e:
            st.warning(f"Content extraction error for {url}: {str(e)}")

        return content_dict

    def scrape_website(self, base_url, max_pages=40, progress_callback=None):
        visited = set()
        all_content = []
        q = deque()
        base_domain = urlparse(base_url).netloc

        for p in PRIORITY_PAGES:
            q.append(urljoin(base_url, p))

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        while q and len(visited) < max_pages:
            url = q.popleft().split("#")[0].split("?")[0]

            if url in visited or not self.is_valid_url(url, base_domain):
                continue

            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    continue

                visited.add(url)
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)

                soup = BeautifulSoup(r.text, "html.parser")
                data = self.extract_content(soup, url)

                if len(data['main_content']) > 100:
                    formatted = f"URL: {data['url']}\nTITLE: {data['title']}\n"
                    if "description" in data['metadata']:
                        formatted += f"DESCRIPTION: {data['metadata']['description']}\n"
                    formatted += f"\nCONTENT:\n{data['main_content']}"
                    all_content.append(formatted)
                    self.scraped_content[url] = data

                for link in soup.find_all("a", href=True):
                    next_url = urljoin(url, link['href']).split("#")[0].split("?")[0]
                    if next_url not in visited and self.is_valid_url(next_url, base_domain):
                        q.append(next_url)

            except Exception:
                continue

        self.status["pages_scraped"] = len(visited)

        if self.company_info['emails'] or self.company_info['phones']:
            header = "COMPANY CONTACT INFORMATION\n" + "="*50 + "\n\n"
            if self.company_info['india_address']:
                header += f"INDIA OFFICE:\n{self.company_info['india_address']}\n\n"
            if self.company_info['singapore_address']:
                header += f"SINGAPORE OFFICE:\n{self.company_info['singapore_address']}\n\n"
            for e in sorted(self.company_info['emails']):
                header += f"Email: {e}\n"
            for p in sorted(self.company_info['phones']):
                header += f"Phone: {p}\n"
            all_content.insert(0, header)

        return ("\n\n" + "="*80 + "\n\n").join(all_content)

    def get_company_contact_info(self):
        info = self.company_info
        if not any([info['emails'], info['phones'], info['india_address'], info['singapore_address']]):
            return "No contact details found."

        msg = "📞 **COMPANY CONTACT INFORMATION**\n\n"
        if info['india_address']:
            msg += f"🇮🇳 **India Office:**\n{info['india_address']}\n\n"
        if info['singapore_address']:
            msg += f"🇸🇬 **Singapore Office:**\n{info['singapore_address']}\n\n"
        if info['emails']:
            msg += "📧 **Emails:**\n" + "\n".join([f"• {e}" for e in info['emails']]) + "\n\n"
        if info['phones']:
            msg += "☎️ **Phones:**\n" + "\n".join([f"• {p}" for p in info['phones']]) + "\n"

        return msg.strip()

    def initialize(self, url, max_pages=40, progress_callback=None):
        try:
            if not OPENROUTER_API_KEY:
                st.error("❌ OPENROUTER_API_KEY not found in environment variables!")
                return False
            
            content = self.scrape_website(url, max_pages, progress_callback)
            if len(content) < 1000:
                st.error("❌ Insufficient content scraped. Check the website URL.")
                return False

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, 
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " "]
            )
            chunks = splitter.split_text(content)

            if len(chunks) == 0:
                st.error("❌ No text chunks created from scraped content.")
                return False

            from sentence_transformers import SentenceTransformer
            
            st.info("📦 Loading embedding model...")
            
            try:
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                st.success("✅ Embedding model loaded!")
            except Exception as e:
                st.error(f"❌ Failed to load embedding model: {str(e)}")
                return False
            
            class CustomEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    try:
                        if not texts:
                            return []
                        embeddings = self.model.encode(
                            texts, 
                            normalize_embeddings=True, 
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        return embeddings.tolist()
                    except Exception as e:
                        st.error(f"Embedding documents error: {str(e)}")
                        raise
                
                def embed_query(self, text):
                    try:
                        embedding = self.model.encode(
                            [text], 
                            normalize_embeddings=True, 
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        return embedding[0].tolist()
                    except Exception as e:
                        st.error(f"Query embedding error: {str(e)}")
                        raise
            
            embeddings = CustomEmbeddings(embedding_model)

            chroma_dir = "./syngrid_chroma"
            if os.path.exists(chroma_dir):
                import shutil
                try:
                    shutil.rmtree(chroma_dir)
                    st.info("🗑️ Cleared old vector database")
                except Exception as e:
                    st.warning(f"Could not remove old chroma directory: {str(e)}")

            st.info("🗄️ Creating vector database...")
            
            try:
                vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    persist_directory=chroma_dir
                )
                st.success("✅ Vector database created!")
            except Exception as e:
                st.error(f"❌ Failed to create vector database: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return False
            
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            self.status["ready"] = True
            return True
            
        except Exception as e:
            st.error(f"❌ Initialization Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def ask(self, question):
        if not self.status["ready"]:
            return "⚠️ Initialization still in progress."

        q_lower = question.lower().strip()

        greetings = ["hi", "hello", "hey", "hai", "hii", "helloo", "hi there", "hello there"]
        if q_lower in greetings:
            return "Hi, I'm Syngrid AI Assistant. How can I assist you?"

        contact_words = ["email", "contact", "phone", "address", "office", "location", "reach", "call"]
        if any(k in q_lower for k in contact_words):
            return self.get_company_contact_info()

        if q_lower in self.cache:
            return self.cache[q_lower]

        try:
            docs = self.retriever.invoke(question)
            if not docs:
                return "I couldn't find relevant information in the scraped data."

            context = "\n\n".join([d.page_content for d in docs])[:4000]

            prompt = f"""Answer the question using the context below. Be concise and helpful.

Context:
{context}

Question: {question}

Answer in 2–4 sentences, focusing on the most relevant information."""

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://syngrid.com",
                "X-Title": "Syngrid AI Assistant"
            }

            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for Syngrid Technologies. Answer questions only using the given context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }

            r = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=60)
            
            if r.status_code == 200:
                ans = r.json()["choices"][0]["message"]["content"].strip()
                self.cache[q_lower] = ans
                return ans
            else:
                return f"⚠️ API Error ({r.status_code}): {r.text[:200]}"

        except requests.exceptions.Timeout:
            return "⚠️ Request timeout. Please try again."
        except requests.exceptions.RequestException as e:
            return f"⚠️ Network error: {str(e)}"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    def save_to_db(self, question, answer):
        db = None
        try:
            db = SessionLocal()
            db.add(ChatHistory(question=question, answer=answer))
            db.commit()
        except Exception as e:
            st.warning(f"Database save error: {str(e)}")
        finally:
            if db:
                db.close()
# ===== ADD THIS TO THE END OF PART 1 CODE =====

# Page Configuration
st.set_page_config(
    page_title="Syngrid AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%); 
    }
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMetric label {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    .header-container { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; 
        border-radius: 20px; 
        margin-bottom: 2rem; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); 
        text-align: center; 
    }
    .header-title { color: white; font-size: 3rem; font-weight: bold; margin: 0; }
    .header-subtitle { color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem; }
    .status-badge { 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-weight: bold; 
        display: inline-block;
        margin: 0.5rem 0;
    }
    .status-ready { background: #10b981; color: white; }
    .status-loading { background: #f59e0b; color: white; }
    
    .contact-form {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "ai" not in st.session_state:
    st.session_state.ai = SyngridAI()
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.question_count = 0
    st.session_state.user_info_collected = False

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 Syngrid AI Assistant</h1>
    <p class="header-subtitle">Powered by Advanced AI | Instant Answers About Syngrid Technologies</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Assistant Status")

    if st.session_state.initialized:
        st.markdown('<div class="status-badge status-ready">✅ Ready</div>', unsafe_allow_html=True)
        st.metric("Pages Scraped", st.session_state.ai.status["pages_scraped"])
        st.metric("Messages", len(st.session_state.messages))
    else:
        st.markdown('<div class="status-badge status-loading">⏳ Initializing…</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂 Download Database")

    if os.path.exists("syngrid_chat.db"):
        with open("syngrid_chat.db", "rb") as f:
            st.download_button(
                "⬇️ Download syngrid_chat.db",
                data=f,
                file_name="syngrid_chat.db",
                mime="application/octet-stream",
                use_container_width=True
            )
    else:
        st.info("📂 Database not created yet.")

    st.markdown("---")
    st.markdown("### 🛠 Quick Tools")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()
    
    if st.button("🔄 Reinitialize AI", use_container_width=True):
        st.session_state.initialized = False
        st.session_state.ai = SyngridAI()
        st.rerun()

# Auto-initialize AI scraper
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing Syngrid AI Assistant…"):
        bar = st.progress(0)
        stat = st.empty()

        def cb(cur, total, url):
            progress = min(cur / total, 1.0)
            bar.progress(progress)
            stat.text(f"📄 Scraping {cur}/{total}: {url[:60]}...")

        ok = st.session_state.ai.initialize(SYNGRID_WEBSITE, max_pages=40, progress_callback=cb)

        if ok:
            st.session_state.initialized = True
            bar.progress(1.0)
            stat.success("✅ Initialization complete!")
            time.sleep(1.5)
            st.rerun()
        else:
            st.error("❌ Initialization failed. Check logs above.")
            st.stop()

# Chat interface
if st.session_state.initialized:

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("💬 Ask anything about Syngrid…"):

        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.question_count += 1

        # Contact form trigger (after 3rd question)
        if st.session_state.question_count == 3 and not st.session_state.user_info_collected:

            # AI response first
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking…"):
                    response = st.session_state.ai.ask(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.ai.save_to_db(user_input, response)

            # Show contact form
            st.markdown("---")
            st.markdown('<div class="contact-form">', unsafe_allow_html=True)
            st.markdown("### 📋 Please Share Your Contact Information")
            st.markdown("*We'd love to stay connected and assist you better!*")

            with st.form("contact_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Full Name *", placeholder="John Doe")
                    email = st.text_input("Email *", placeholder="john@example.com")
                with col2:
                    phone = st.text_input("Phone *", placeholder="+91 98765 43210")
                    st.markdown("<br>", unsafe_allow_html=True)

                submit = st.form_submit_button("✅ Submit & Continue", use_container_width=True)

                if submit:
                    if name and email and phone:
                        db = None
                        try:
                            db = SessionLocal()
                            db.add(UserContact(name=name, email=email, phone=phone))
                            db.commit()

                            st.session_state.user_info_collected = True
                            st.success("✅ Information saved successfully! You can continue chatting.")
                            time.sleep(1.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to save: {str(e)}")
                        finally:
                            if db:
                                db.close()
                    else:
                        st.error("⚠️ Please fill all required fields.")

            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        # Normal conversation response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking…"):
                answer = st.session_state.ai.ask(user_input)
                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.ai.save_to_db(user_input, answer)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.8; padding: 20px; color: white;'>
    🤖 <strong>Syngrid AI Assistant</strong> — Powered by Streamlit & OpenRouter  
    <br>© 2025 Syngrid Technologies. All rights reserved.
</div>
""", unsafe_allow_html=True)
