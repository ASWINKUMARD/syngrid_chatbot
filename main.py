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
import re
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
# STUNNING CUSTOM CSS
# ========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientWave 20s ease infinite;
    }
    
    @keyframes gradientWave {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .syngrid-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 25px 70px rgba(102, 126, 234, 0.5), 0 0 120px rgba(118, 75, 162, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
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
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        animation: headerShine 4s infinite;
    }
    
    @keyframes headerShine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .syngrid-logo {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #a8edea 50%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        position: relative;
        z-index: 1;
        filter: drop-shadow(0 0 30px rgba(255, 255, 255, 0.5));
    }
    
    .syngrid-tagline {
        color: #e0f2fe;
        font-size: 1.4rem;
        font-weight: 300;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 30px 30px 8px 30px;
        margin: 1.8rem 0;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5);
        font-size: 1.15rem;
        animation: slideInRight 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        border: 2px solid rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
        line-height: 1.6;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent);
        transition: left 0.6s;
    }
    
    .user-message:hover::before {
        left: 100%;
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(240, 248, 255, 0.98) 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 30px 30px 30px 8px;
        margin: 1.8rem 0;
        color: #1e293b;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        border-left: 6px solid #667eea;
        font-size: 1.15rem;
        animation: slideInLeft 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        line-height: 1.8;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(60px) scale(0.85); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-60px) scale(0.85); }
        to { opacity: 1; transform: translateX(0) scale(1); }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-weight: 700;
        font-size: 1.15rem;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        border: 2px solid rgba(255, 255, 255, 0.25);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-6px) scale(1.08);
        box-shadow: 0 18px 50px rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 50%, #f093fb 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-3px) scale(1.03);
    }
    
    .stTextInput>div>div>input {
        border-radius: 50px;
        border: 3px solid rgba(102, 126, 234, 0.5);
        padding: 1.3rem 2.2rem;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        font-size: 1.15rem;
        color: #f0f9ff;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #a8edea;
        box-shadow: 0 0 40px rgba(168, 237, 234, 0.5);
        background: rgba(255, 255, 255, 0.12);
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #cbd5e1;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #a8edea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0f2fe !important;
        font-weight: 600;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.08) 100%);
        padding: 1.8rem;
        border-radius: 25px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stMetric:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(255, 255, 255, 0.3);
    }
    
    h1, h2, h3 { 
        background: linear-gradient(135deg, #fff 0%, #a8edea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    p, label { 
        color: #f0f9ff !important;
        line-height: 1.7;
    }
    
    .welcome-container {
        text-align: center;
        padding: 5rem 3rem;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-radius: 30px;
        border: 2px solid rgba(255, 255, 255, 0.25);
        margin: 2rem 0;
        backdrop-filter: blur(15px);
    }
    
    .welcome-container h3 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    .welcome-container p {
        font-size: 1.3rem;
        color: #e0f2fe !important;
        line-height: 1.8;
    }
    
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .modal-content {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.98) 0%, rgba(118, 75, 162, 0.98) 100%);
        padding: 3rem;
        border-radius: 30px;
        max-width: 600px;
        width: 90%;
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.5);
        border: 2px solid rgba(255, 255, 255, 0.3);
        animation: slideUp 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    
    @keyframes slideUp {
        from { transform: translateY(100px) scale(0.9); opacity: 0; }
        to { transform: translateY(0) scale(1); opacity: 1; }
    }
    
    .modal-content h2 {
        color: white !important;
        background: none !important;
        -webkit-text-fill-color: white !important;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .modal-content p {
        color: #e0f2fe !important;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .init-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.08) 100%);
        padding: 3.5rem;
        border-radius: 30px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
        backdrop-filter: blur(15px);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    .stSpinner > div {
        border-top-color: #a8edea !important;
    }
    
    ::-webkit-scrollbar { width: 16px; }
    ::-webkit-scrollbar-track { 
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 3px solid rgba(0, 0, 0, 0.3);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #f093fb 100%);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        margin: 2.5rem 0;
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# DATABASE SETUP
# ========================================
USE_MEMORY_DB = os.environ.get("USE_MEMORY_DB", "true").lower() == "true"
DATABASE_URL = "sqlite:///:memory:" if USE_MEMORY_DB else "sqlite:///./syngrid_chat.db"
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
    phone = Column(String(30), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

# ========================================
# VALIDATION FUNCTIONS
# ========================================
def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number (8-15 digits, flexible for international)"""
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone)
    return 8 <= len(cleaned) <= 15 and cleaned.replace('+', '').isdigit()

def validate_name(name):
    """Validate name (at least 2 characters)"""
    return len(name.strip()) >= 2 and re.match(r'^[a-zA-Z\s]+$', name.strip()) is not None

# ========================================
# CONFIGURATION
# ========================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SYNGRID_WEBSITE = "https://syngrid.com/"

PRIORITY_PAGES = [
    "", "about", "about-us", "services", "solutions", "products",
    "contact", "contact-us", "team", "careers", "footer", "reach-us"
]

# ========================================
# SIMPLE EMBEDDINGS CLASS
# ========================================
class SimpleEmbeddings(Embeddings):
    """Hash-based embeddings - no external models needed"""
    
    def __init__(self):
        self.dimension = 384
        
    def _tokenize(self, text: str) -> List[str]:
        words = text.lower().split()
        words = [''.join(c for c in word if c.isalnum()) for word in words]
        return [w for w in words if len(w) > 2]
    
    def _get_vector(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.dimension
        
        for i, token in enumerate(tokens[:25]):
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            positions = [(hash_val + j) % self.dimension for j in range(15)]
            
            for pos in positions:
                vector[pos] += 1.0 / (i + 1)
        
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
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'india_address': None,
            'singapore_address': None,
        }

    def extract_contact_info(self, soup, text, url):
        """Extract emails, phones, and addresses"""
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif')):
                self.company_info['emails'].add(email.lower())

        # Extract phones (flexible international format)
        phone_patterns = [
            r'\+\d{1,3}[\s\-]?\d{4,5}[\s\-]?\d{4,5}',
            r'\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{4}',
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if 8 <= len(cleaned) <= 15:
                    self.company_info['phones'].add(phone.strip())

        # Extract addresses from contact/footer pages
        if any(kw in url.lower() for kw in ['contact', 'footer', 'about', 'reach']):
            lines = text.split('\n')
            
            # India address
            if not self.company_info['india_address']:
                for i, line in enumerate(lines):
                    if ('madurai' in line.lower() or 'tbi' in line.lower()) and '625' in line:
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        address_parts = [lines[j].strip() for j in range(start, end) 
                                       if lines[j].strip() and len(lines[j].strip()) > 5]
                        if address_parts:
                            full_address = ' '.join(address_parts)
                            if 'madurai' in full_address.lower() and 30 < len(full_address) < 300:
                                self.company_info['india_address'] = full_address
                                break
            
            # Singapore address
            if not self.company_info['singapore_address']:
                for i, line in enumerate(lines):
                    if 'singapore' in line.lower() and re.search(r'\d{6}', line):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        address_parts = [lines[j].strip() for j in range(start, end) 
                                       if lines[j].strip() and len(lines[j].strip()) > 5]
                        if address_parts:
                            full_address = ' '.join(address_parts)
                            if 'singapore' in full_address.lower() and 30 < len(full_address) < 300:
                                self.company_info['singapore_address'] = full_address
                                break

    def scrape_website(self, base_url, max_pages=25, progress_callback=None):
        """Scrape website with enhanced content extraction"""
        visited = set()
        all_content = []
        queue = deque()
        base_domain = urlparse(base_url).netloc

        # Add priority pages
        for page in PRIORITY_PAGES:
            queue.append(urljoin(base_url, page))

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        while queue and len(visited) < max_pages:
            current_url = queue.popleft()
            current_url = current_url.split('#')[0].split('?')[0]

            if current_url in visited or len(visited) >= max_pages:
                continue

            try:
                resp = requests.get(current_url, headers=headers, timeout=15, allow_redirects=True)
                if resp.status_code != 200:
                    continue

                visited.add(current_url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages)

                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'aside', 'iframe']):
                    tag.decompose()

                text = soup.get_text(separator='\n', strip=True)
                self.extract_contact_info(soup, text, current_url)

                lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 20]
                clean_text = '\n'.join(lines)

                if len(clean_text) > 200:
                    all_content.append(f"SOURCE: {current_url}\n\n{clean_text}")

                # Find more links
                if len(visited) < max_pages:
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(current_url, link['href'])
                        next_url = next_url.split('#')[0].split('?')[0]
                        if (urlparse(next_url).netloc == base_domain and 
                            next_url not in visited and next_url not in queue):
                            queue.append(next_url)

            except Exception:
                continue

        self.status["pages_scraped"] = len(visited)

        # Add contact info to content
        contact_info = self.format_company_info()
        if contact_info:
            all_content.insert(0, contact_info)

        return "\n\n=== PAGE BREAK ===\n\n".join(all_content)

    def format_company_info(self):
        """Format contact information"""
        if not any([self.company_info['emails'], self.company_info['phones'],
                   self.company_info['india_address'], self.company_info['singapore_address']]):
            return ""

        info = "COMPANY CONTACT INFORMATION\n" + "="*50 + "\n\n"
        
        if self.company_info['india_address']:
            info += f"INDIA OFFICE:\n{self.company_info['india_address']}\n\n"
        
        if self.company_info['singapore_address']:
            info += f"SINGAPORE OFFICE:\n{self.company_info['singapore_address']}\n\n"
        
        if self.company_info['emails']:
            info += "EMAILS:\n" + "\n".join(f"  • {e}" for e in sorted(self.company_info['emails'])) + "\n\n"
        
        if self.company_info['phones']:
            info += "PHONES:\n" + "\n".join(f"  • {p}" for p in sorted(self.company_info['phones']))
        
        return info

    def get_contact_info(self):
        """Return formatted contact information"""
        if not any([self.company_info['emails'], self.company_info['phones'],
                   self.company_info['india_address'], self.company_info['singapore_address']]):
            return "📞 Contact information not available. Please visit the website."

        response = "📞 SYNGRID CONTACT INFORMATION\n" + "="*50 + "\n\n"
        
        if self.company_info['india_address'] or self.company_info['singapore_address']:
            response += "🏢 OFFICES:\n\n"
            if self.company_info['india_address']:
                response += f"📍 India Office:\n   {self.company_info['india_address']}\n\n"
            if self.company_info['singapore_address']:
                response += f"📍 Singapore Office:\n   {self.company_info['singapore_address']}\n\n"
        
        if self.company_info['emails']:
            response += "📧 EMAIL:\n" + "\n".join(f"   • {e}" for e in sorted(self.company_info['emails'])) + "\n\n"
        
        if self.company_info['phones']:
            response += "☎️ PHONE:\n" + "\n".join(f"   • {p}" for p in sorted(self.company_info['phones']))
        
        return response.strip()

    def initialize(self, url, max_pages, progress_callback=None):
        """Initialize AI Assistant"""
        try:
            if not OPENROUTER_API_KEY:
                self.status["message"] = "API key not configured"
                return False
            
            self.status["message"] = "🔍 Analyzing website..."
            content = self.scrape_website(url, max_pages, progress_callback)

            if len(content) < 500:
                self.status["message"] = "❌ Insufficient content"
                return False

            self.status["message"] = "📝 Processing content..."
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150,
                separators=["\n\n=== PAGE BREAK ===\n\n", "\n\n", "\n", " "]
            )
            chunks = splitter.split_text(content)

            self.status["message"] = "🧠 Building knowledge base..."
            embeddings = SimpleEmbeddings()
            self.vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            self.status["ready"] = True
            self.status["message"] = "✅ Ready!"
            return True

        except Exception as e:
            self.status["message"] = f"❌ Error: {str(e)}"
            return False

    def ask(self, question):
        """Ask question to AI"""
        if not self.status["ready"]:
            return "⚡ Please initialize the assistant first."

        if not OPENROUTER_API_KEY:
            return "⚠️ API key not configured."

        q_lower = question.lower().strip()
        
        # Check for contact-related queries
        contact_keywords = ['contact', 'email', 'phone', 'address', 'office', 
                          'location', 'reach', 'call', 'visit', 'where']
        if any(kw in q_lower for kw in contact_keywords):
            return self.get_contact_info()
        
        # Check cache
        if q_lower in self.cache:
            return self.cache[q_lower]

        try:
            docs = self.retriever.invoke(question)
            if not docs:
                return "I couldn't find relevant information. Please rephrase your question."

            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            prompt = f"""You are Syngrid AI Assistant. Answer based on the context about Syngrid Technologies.

Context:
{context[:3500]}

Question: {question}

Provide a clear, helpful answer (2-4 sentences):"""

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
                
                # Save to database
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
                return "⚠️ Having trouble processing that. Please try again."

        except Exception as e:
            return f"⚠️ Error: {str(e)[:100]}"

# ========================================
# SESSION STATE
# ========================================
if 'ai' not in st.session_state:
    st.session_state.ai = SyngridAI()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'user_info_collected' not in st.session_state:
    st.session_state.user_info_collected = False
if 'show_user_form' not in st.session_state:
    st.session_state.show_user_form = False

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
# USER INFO COLLECTION MODAL
# ========================================
if st.session_state.show_user_form and not st.session_state.user_info_collected:
    with st.container():
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%); 
                    padding: 3rem; border-radius: 30px; border: 2px solid rgba(255, 255, 255, 0.3);
                    backdrop-filter: blur(15px); margin: 2rem 0;'>
            <h2 style='text-align: center; color: white !important; background: none !important; 
                       -webkit-text-fill-color: white !important; margin-bottom: 1rem;'>
                📝 Please Share Your Contact Information
            </h2>
            <p style='text-align: center; color: #e0f2fe !important; margin-bottom: 2rem; font-size: 1.1rem;'>
                To continue using the assistant, we need your details
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("user_info_form"):
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
            phone = st.text_input("Phone Number * (8-15 digits, international format accepted)", 
                                 placeholder="+1234567890 or 1234567890")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submitted = st.form_submit_button("✅ Submit", use_container_width=True)
            
            if submitted:
                errors = []
                if not validate_name(name):
                    errors.append("❌ Name must be at least 2 characters (letters only)")
                if not validate_email(email):
                    errors.append("❌ Please enter a valid email address")
                if not validate_phone(phone):
                    errors.append("❌ Phone must be 8-15 digits (international formats accepted)")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    try:
                        db = SessionLocal()
                        contact = UserContact(
                            name=name.strip(),
                            email=email.strip(),
                            phone=phone.strip(),
                            timestamp=datetime.now(timezone.utc)
                        )
                        db.add(contact)
                        db.commit()
                        db.close()
                        
                        st.session_state.user_info_collected = True
                        st.session_state.show_user_form = False
                        st.success("✅ Thank you! Your information has been saved.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Database error: {str(e)}")

# ========================================
# INITIALIZATION SECTION
# ========================================
if not st.session_state.initialized:
    st.markdown('<div class="init-section">', unsafe_allow_html=True)
    st.markdown("### 🚀 Welcome to Syngrid AI Assistant")
    st.markdown("Initialize the AI assistant to start exploring Syngrid Technologies' innovative solutions, services, and expertise.")
    
    if not OPENROUTER_API_KEY:
        st.error("⚠️ **API Key Required**: Configure OPENROUTER_API_KEY in Streamlit secrets.")
        st.info("📖 **Setup**: Go to Settings → Secrets and add: `OPENROUTER_API_KEY = \"your-key\"`")
    
    st.markdown("")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("⚡ Initialize Assistant", use_container_width=True, key="init_btn", 
                    disabled=not OPENROUTER_API_KEY):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.markdown(
                    f"<p style='text-align:center; color:#e0f2fe; font-size:1.2rem'>⚡ Analyzing: {current}/{total} pages</p>",
                    unsafe_allow_html=True
                )
            
            with st.spinner("🔮 Initializing Syngrid AI..."):
                success = st.session_state.ai.initialize(
                    SYNGRID_WEBSITE,
                    25,
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
        st.metric("💬 Messages", len(st.session_state.messages))
    
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
                    <strong>Services • Solutions • Technologies • Contact Info • Projects</strong>
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
            placeholder="💭 Ask about services, technologies, contact information...",
            label_visibility="collapsed"
        )
    
    with col2:
        send = st.button("⚡ Send", use_container_width=True)
    
    if send and user_input and user_input.strip():
        # Check if need to collect user info
        st.session_state.question_count += 1
        
        if st.session_state.question_count == 3 and not st.session_state.user_info_collected:
            # Answer the question first
            with st.spinner("⚡ Thinking..."):
                response = st.session_state.ai.ask(user_input.strip())
                st.session_state.messages.append((user_input.strip(), response))
            
            # Then show the form
            st.session_state.show_user_form = True
            st.rerun()
        else:
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
            st.session_state.question_count = 0
            st.rerun()
    
    with col3:
        if st.button("⚙️ Restart Assistant", use_container_width=True):
            st.session_state.messages = []
            st.session_state.initialized = False
            st.session_state.ai = SyngridAI()
            st.session_state.question_count = 0
            st.session_state.user_info_collected = False
            st.session_state.show_user_form = False
            st.rerun()

# ========================================
# FOOTER
# ========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #e0f2fe; font-size: 0.95rem; padding: 2rem;'>
    ⚡ Powered by Syngrid Technologies | AI-Driven Innovation<br>
    <small style='color: #cbd5e1;'>Secure • Intelligent • Always Learning</small>
</div>
""", unsafe_allow_html=True)
