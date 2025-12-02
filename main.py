import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import re
import os
import time
import sys

# Fix for Streamlit Cloud sqlite3 issue
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Page Configuration
st.set_page_config(
    page_title="Syngrid AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for Stunning UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated gradient background */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism chat container */
    .stChatFloatingInputContainer {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    
    /* Stunning header with animation */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .header-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        animation: titleFloat 3s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    .header-tagline {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.8rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Status badges with glow */
    .status-badge {
        display: inline-block;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .status-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    .status-ready {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        animation: pulseGlow 2s infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
        50% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.8); }
    }
    
    .status-loading {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    /* Enhanced chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* User message styling */
    [data-testid="stChatMessageContent"] {
        color: white;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* Metrics with gradient */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.15);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Form styling */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.4);
        background: white;
    }
    
    /* Contact form */
    .contact-form {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.9));
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Info boxes with gradient border */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        color: white;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Chat input styling */
    .stChatInputContainer > div {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 25px;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border-left: 4px solid;
        padding: 1rem;
    }
    
    /* Feature list styling */
    .feature-item {
        padding: 0.5rem 0;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateX(5px);
        color: white;
    }
    
    /* Footer styling */
    .footer-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Icon styling */
    .icon-wrapper {
        display: inline-block;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

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
SYNGRID_WEBSITE = "https://syngrid.com/"
USE_AI_API = False
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.2-3b-instruct:free"

PRIORITY_PAGES = [
    "", "about", "about-us", "services", "solutions", "products", 
    "contact", "contact-us", "team", "careers", "blog", "case-studies",
    "portfolio", "industries", "technology", "expertise", "what-we-do",
    "who-we-are", "footer", "locations", "reach-us"
]

class SyngridAI:
    def __init__(self):
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "message": "Not initialized", "pages_scraped": 0}
        self.scraped_content = {}
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'india_address': None,
            'singapore_address': None,
            'social_media': {}
        }

    def clean_address(self, text):
        text = ' '.join(text.split())
        text = re.sub(r'(Corporate Office|Branch Office|Head Office|Registered Office)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_contact_info(self, soup, text, url):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif')):
                self.company_info['emails'].add(email.lower())

        phone_patterns = [
            r'\+91\s*\d{5}\s*\d{5}',
            r'\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        ]

        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if 10 <= len(cleaned) <= 15:
                    self.company_info['phones'].add(phone.strip())

        if any(keyword in url.lower() for keyword in ['contact', 'footer', 'about', 'reach']):
            lines = text.split('\n')

            if not self.company_info['india_address']:
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if ('tbi' in line_lower or 'madurai' in line_lower) and '625' in line:
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        address_parts = []
                        for j in range(start, end):
                            current_line = lines[j].strip()
                            if current_line and len(current_line) > 5:
                                if not any(skip in current_line.lower() for skip in ['career', 'job', 'hiring', 'vacancy']):
                                    address_parts.append(current_line)
                        if address_parts:
                            full_address = ' '.join(address_parts)
                            if 'madurai' in full_address.lower() and len(full_address) > 30:
                                address_clean = self.clean_address(full_address)
                                if 20 < len(address_clean) < 300:
                                    self.company_info['india_address'] = address_clean
                                    break

            if not self.company_info['singapore_address']:
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if 'singapore' in line_lower and (re.search(r'\d{6}', line) or 'road' in line_lower):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        address_parts = []
                        for j in range(start, end):
                            current_line = lines[j].strip()
                            if current_line and len(current_line) > 5:
                                if not any(skip in current_line.lower() for skip in ['career', 'job', 'hiring']):
                                    address_parts.append(current_line)
                        if address_parts:
                            full_address = ' '.join(address_parts)
                            if 'singapore' in full_address.lower() and len(full_address) > 30:
                                address_clean = self.clean_address(full_address)
                                if 20 < len(address_clean) < 300:
                                    self.company_info['singapore_address'] = address_clean
                                    break

    def is_valid_url(self, url, base_domain):
        parsed = urlparse(url)
        if parsed.netloc != base_domain:
            return False
        skip_patterns = [
            r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$',
            r'/wp-admin/', r'/wp-content/(?!.*contact)', r'/wp-includes/',
            r'/feed/', r'/rss/', r'/sitemap', r'/login', r'/register',
            r'\.css$', r'\.js$', r'/cart/', r'/checkout/'
        ]
        for pattern in skip_patterns:
            if re.search(pattern, url.lower()):
                return False
        return True

    def extract_content(self, soup, url):
        content_dict = {'url': url, 'title': '', 'main_content': '', 'metadata': {}}
        
        title_tag = soup.find('title')
        if title_tag:
            content_dict['title'] = title_tag.get_text(strip=True)

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content_dict['metadata']['description'] = meta_desc.get('content')

        full_text = soup.get_text(separator='\n', strip=True)
        self.extract_contact_info(soup, full_text, url)

        for tag in soup(['script', 'style', 'nav', 'aside', 'iframe', 'noscript', 'form', 'button']):
            tag.decompose()

        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '#content', '#main', '.page-content', '.entry-content', '.post-content',
            'footer', '.footer', '#footer'
        ]

        content_parts = []
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                content_parts.append(element)

        if not content_parts:
            main_content = soup.find('body')
        else:
            main_content = soup.new_tag('div')
            for part in content_parts:
                main_content.append(part)

        if main_content:
            headings = []
            for heading in main_content.find_all(['h1', 'h2', 'h3', 'h4']):
                heading_text = heading.get_text(strip=True)
                if len(heading_text) > 3:
                    headings.append(heading_text)

            text = main_content.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            meaningful_lines = []
            seen = set()
            for line in lines:
                if len(line) > 20 and line not in seen:
                    meaningful_lines.append(line)
                    seen.add(line)

            content_dict['main_content'] = '\n'.join(meaningful_lines)
            content_dict['metadata']['headings'] = headings

        return content_dict

    def scrape_website(self, base_url, max_pages=40, progress_callback=None):
        visited = set()
        all_content = []
        queue = deque()
        base_domain = urlparse(base_url).netloc

        for page in PRIORITY_PAGES:
            priority_url = urljoin(base_url, page)
            queue.append(priority_url)

        if base_url not in queue:
            queue.append(base_url)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        while queue and len(visited) < max_pages:
            current_url = queue.popleft()
            current_url = current_url.split('#')[0].split('?')[0]

            if current_url in visited or len(visited) >= max_pages:
                continue

            if not self.is_valid_url(current_url, base_domain):
                continue

            try:
                resp = requests.get(current_url, headers=headers, timeout=15, allow_redirects=True)
                if resp.status_code != 200:
                    continue

                visited.add(current_url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages, current_url)

                soup = BeautifulSoup(resp.text, 'html.parser')
                content_data = self.extract_content(soup, current_url)

                if len(content_data['main_content']) > 100:
                    formatted_content = f"""URL: {content_data['url']}\nTITLE: {content_data['title']}\n"""
                    if content_data['metadata'].get('description'):
                        formatted_content += f"DESCRIPTION: {content_data['metadata']['description']}\n"
                    if content_data['metadata'].get('headings'):
                        formatted_content += f"\nKEY SECTIONS: {', '.join(content_data['metadata']['headings'][:5])}\n"
                    formatted_content += f"\nCONTENT:\n{content_data['main_content']}"
                    all_content.append(formatted_content)
                    self.scraped_content[current_url] = content_data

                if len(visited) < max_pages:
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(current_url, link['href'])
                        next_url = next_url.split('#')[0].split('?')[0]
                        if (next_url not in visited and next_url not in queue and 
                            self.is_valid_url(next_url, base_domain)):
                            queue.append(next_url)

            except Exception as e:
                continue

        self.status["pages_scraped"] = len(visited)
        contact_info_text = self.format_company_info()
        if contact_info_text:
            all_content.insert(0, contact_info_text)

        separator = "=" * 80
        return f"\n\n{separator}\n\n".join(all_content)

    def format_company_info(self):
        if not any([self.company_info['emails'], self.company_info['phones'],
                   self.company_info['india_address'], self.company_info['singapore_address']]):
            return ""

        info_text = "COMPANY CONTACT INFORMATION\n" + "="*50 + "\n\n"

        if self.company_info['india_address']:
            info_text += f"INDIA OFFICE (Corporate Office):\n  {self.company_info['india_address']}\n\n"

        if self.company_info['singapore_address']:
            info_text += f"SINGAPORE OFFICE:\n  {self.company_info['singapore_address']}\n\n"

        if self.company_info['emails']:
            info_text += "EMAIL ADDRESSES:\n"
            for email in sorted(self.company_info['emails']):
                info_text += f"  • {email}\n"
            info_text += "\n"

        if self.company_info['phones']:
            info_text += "PHONE NUMBERS:\n"
            for phone in sorted(self.company_info['phones']):
                info_text += f"  • {phone}\n"
            info_text += "\n"

        return info_text

    def get_company_contact_info(self):
        if not any([self.company_info['emails'], self.company_info['phones'],
                   self.company_info['india_address'], self.company_info['singapore_address']]):
            return "Contact information not found. Please visit the company website."

        response = "📞 **COMPANY CONTACT INFORMATION**\n\n"

        if self.company_info['india_address'] or self.company_info['singapore_address']:
            response += "🏢 **OFFICE ADDRESSES:**\n\n"
            if self.company_info['india_address']:
                response += f"📍 **India Office (Corporate Office):**\n   {self.company_info['india_address']}\n\n"
            if self.company_info['singapore_address']:
                response += f"📍 **Singapore Office:**\n   {self.company_info['singapore_address']}\n\n"

        if self.company_info['emails']:
            response += "📧 **EMAIL:**\n"
            for email in sorted(self.company_info['emails']):
                response += f"   • {email}\n"
            response += "\n"

        if self.company_info['phones']:
            response += "☎️ **PHONE:**\n"
            for phone in sorted(self.company_info['phones']):
                response += f"   • {phone}\n"

        return response.strip()

    def initialize(self, url, max_pages=40, progress_callback=None):
        try:
            content = self.scrape_website(url, max_pages, progress_callback)
            if len(content) < 1000:
                return False

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " "]
            )
            chunks = splitter.split_text(content)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            vectorstore = Chroma.from_texts(
                chunks,
                embedding=embeddings,
                persist_directory="./syngrid_chroma"
            )

            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            self.status["ready"] = True
            return True

        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")
            return False

    def extract_answer_from_context(self, question, context):
        """Smart context extraction when API is unavailable"""
        q_lower = question.lower()
        
        sentences = []
        for chunk in context.split('\n'):
            chunk = chunk.strip()
            if len(chunk) > 50:
                sentences.append(chunk)
        
        relevant_sentences = []
        question_words = set(q_lower.split())
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for word in question_words if len(word) > 3 and word in sentence_lower)
            if matches > 0:
                relevant_sentences.append((matches, sentence))
        
        relevant_sentences.sort(reverse=True, key=lambda x: x[0])
        
        if relevant_sentences:
            top_sentences = [sent[1] for sent in relevant_sentences[:3]]
            answer = " ".join(top_sentences)
            
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            return answer
        else:
            preview = " ".join(sentences[:2])
            if len(preview) > 400:
                preview = preview[:400] + "..."
            return preview if preview else "I found information but couldn't extract a specific answer. Please try rephrasing your question."

    def ask(self, question):
        if not self.status["ready"]:
            return "⚠️ Please wait for initialization to complete."

        q_lower = question.lower().strip()
        contact_keywords = ['contact', 'email', 'phone', 'address', 'office', 'location',
                          'reach', 'call', 'visit', 'headquarters', 'where']

        if any(keyword in q_lower for keyword in contact_keywords):
            return self.get_company_contact_info()

        if q_lower in self.cache:
            return self.cache[q_lower]

        try:
            docs = self.retriever.invoke(question)
            if not docs:
                return "I couldn't find relevant information. Could you rephrase?"

            context = "\n\n".join([doc.page_content for doc in docs])

            if not USE_AI_API or not OPENROUTER_API_KEY:
                answer = self.extract_answer_from_context(question, context)
                self.cache[q_lower] = answer
                return answer

            prompt = f"""You are Syngrid AI Assistant. Answer based on the provided context about Syngrid Technologies.

Context:
{context[:4000]}

Question: {question}

Provide a helpful, accurate answer in 2-4 sentences."""

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://syngrid.com",
                "X-Title": "Syngrid AI Assistant"
            }

            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are Syngrid AI Assistant. Answer questions about Syngrid Technologies based only on the given context. Be concise and helpful."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }

            response = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                if answer and len(answer) > 10:
                    self.cache[q_lower] = answer
                    return answer
                else:
                    return self.extract_answer_from_context(question, context)
            else:
                return self.extract_answer_from_context(question, context)

        except Exception as e:
            try:
                docs = self.retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                return self.extract_answer_from_context(question, context)
            except:
                return f"⚠️ Error: {str(e)}"

    def save_to_db(self, question, answer):
        try:
            db = SessionLocal()
            chat_entry = ChatHistory(question=question, answer=answer, timestamp=datetime.now(timezone.utc))
            db.add(chat_entry)
            db.commit()
            db.close()
        except Exception as e:
            pass

# Initialize session state
if 'ai' not in st.session_state:
    st.session_state.ai = SyngridAI()
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.question_count = 0
    st.session_state.user_info_collected = False

# Stunning Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">
        <span class="icon-wrapper">🤖</span> Syngrid AI Assistant
    </h1>
    <p class="header-subtitle">Powered by Advanced AI & Machine Learning</p>
    <p class="header-tagline">✨ Instant Answers | Smart Context Matching | 40+ Pages Knowledge Base</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Status Section
    st.markdown("### ⚡ Assistant Status")
    
    if st.session_state.initialized:
        st.markdown('<div class="status-badge status-ready">✅ Online & Ready</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Pages", st.session_state.ai.status["pages_scraped"])
        with col2:
            st.metric("📧 Emails", len(st.session_state.ai.company_info['emails']))
        with col3:
            st.metric("💬 Chats", len(st.session_state.messages))
    else:
        st.markdown('<div class="status-badge status-loading pulse">🔄 Initializing AI...</div>', unsafe_allow_html=True)
        st.info("⏳ Scraping website and building knowledge base...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # AI Configuration
    st.markdown("### 🔑 AI Enhancement (Optional)")
    st.markdown("**Currently using: Smart Context Matching** ✨")
    
    with st.expander("🚀 Upgrade to Advanced AI"):
        st.markdown("""
        <div class="glass-card">
        <p style='font-size: 0.9rem;'>For even better responses, add a free API key:</p>
        <ul style='font-size: 0.85rem;'>
            <li>🔗 <a href='https://openrouter.ai/keys' target='_blank'>OpenRouter</a> - Free tier available</li>
            <li>⚡ <a href='https://console.groq.com' target='_blank'>Groq</a> - Lightning fast & free</li>
            <li>🎯 <a href='https://api.together.xyz' target='_blank'>Together AI</a> - Free credits</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        api_key_input = st.text_input("API Key (Optional)", type="password", placeholder="sk-or-v1-...")
        
        if api_key_input:
            global OPENROUTER_API_KEY, USE_AI_API
            OPENROUTER_API_KEY = api_key_input
            USE_AI_API = True
            st.success("✅ Advanced AI Enabled!")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.question_count = 0
            st.rerun()
    
    with col2:
        if st.button("📞 Contact", use_container_width=True):
            if st.session_state.initialized:
                contact_info = st.session_state.ai.get_company_contact_info()
                st.info(contact_info)
    
    st.markdown("---")
    
    # Features Section
    st.markdown("### ✨ Features")
    st.markdown("""
    <div class="glass-card">
        <div class="feature-item">⚡ Lightning-fast responses</div>
        <div class="feature-item">🧠 Smart AI matching</div>
        <div class="feature-item">📚 40+ pages knowledge</div>
        <div class="feature-item">💾 Auto conversation save</div>
        <div class="feature-item">🔒 Secure & private</div>
        <div class="feature-item">🎨 Beautiful interface</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About Section
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div class="glass-card" style='font-size: 0.85rem;'>
        <p><strong>Syngrid AI Assistant</strong> uses advanced Natural Language Processing to provide instant, accurate answers about Syngrid Technologies.</p>
        <p style='margin-top: 0.5rem;'>Built with ❤️ using Streamlit, LangChain & HuggingFace.</p>
    </div>
    """, unsafe_allow_html=True)

# Auto-initialize on first load
if not st.session_state.initialized:
    st.markdown("""
    <div class="glass-card" style='text-align: center; padding: 2rem;'>
        <h3 style='color: white; margin-bottom: 1rem;'>🚀 Initializing Syngrid AI Assistant</h3>
        <p style='color: rgba(255,255,255,0.8);'>Please wait while we scrape and process the Syngrid website...</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner(""):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, url):
            progress = current / total
            progress_bar.progress(progress)
            status_text.markdown(f"""
            <div class="glass-card">
                <p style='color: white; margin: 0;'>📄 Scraping page <strong>{current}/{total}</strong></p>
                <p style='color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0.5rem 0 0 0;'>{url[:60]}...</p>
            </div>
            """, unsafe_allow_html=True)
        
        success = st.session_state.ai.initialize(SYNGRID_WEBSITE, max_pages=40, progress_callback=update_progress)
        
        if success:
            st.session_state.initialized = True
            progress_bar.progress(1.0)
            status_text.markdown("""
            <div class="glass-card" style='text-align: center;'>
                <h3 style='color: #10b981; margin: 0;'>✅ Initialization Complete!</h3>
                <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;'>Ready to answer your questions</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Failed to initialize. Please refresh the page.")
            st.stop()

# Chat Interface
if st.session_state.initialized:
    # Welcome message on first load
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="glass-card" style='margin: 2rem 0;'>
            <h3 style='color: white; margin-bottom: 1rem;'>👋 Welcome to Syngrid AI Assistant!</h3>
            <p style='color: rgba(255,255,255,0.9); margin-bottom: 0.5rem;'>I'm here to help you learn about Syngrid Technologies. You can ask me:</p>
            <ul style='color: rgba(255,255,255,0.8);'>
                <li>📞 Contact information and office locations</li>
                <li>💼 Services and solutions offered</li>
                <li>🏢 Company information and expertise</li>
                <li>👥 Team and career opportunities</li>
                <li>🔧 Technologies and industries served</li>
            </ul>
            <p style='color: rgba(255,255,255,0.9); margin-top: 1rem;'><strong>Try asking:</strong> "What services does Syngrid offer?" or "How can I contact Syngrid?"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about Syngrid Technologies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        st.session_state.question_count += 1
        
        # Show contact form after 3 questions
        if st.session_state.question_count == 3 and not st.session_state.user_info_collected:
            with st.chat_message("assistant", avatar="🤖"):
                response = st.session_state.ai.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.ai.save_to_db(prompt, response)
            
            st.markdown("---")
            st.markdown("""
            <div class="contact-form">
                <h3 style='text-align: center; color: #667eea; margin-bottom: 1.5rem;'>📋 Continue Chatting</h3>
                <p style='text-align: center; color: #666; margin-bottom: 1.5rem;'>Please provide your contact information to continue using the assistant</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("contact_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("👤 Full Name *", placeholder="John Doe")
                    phone = st.text_input("📱 Phone Number *", placeholder="+91 98765 43210")
                
                with col2:
                    email = st.text_input("📧 Email Address *", placeholder="john@example.com")
                    company = st.text_input("🏢 Company (Optional)", placeholder="Your Company")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submitted = st.form_submit_button("✅ Submit & Continue Chatting", use_container_width=True)
                
                if submitted:
                    if name and email and phone:
                        try:
                            db = SessionLocal()
                            contact = UserContact(name=name, email=email, phone=phone, timestamp=datetime.now(timezone.utc))
                            db.add(contact)
                            db.commit()
                            db.close()
                            st.session_state.user_info_collected = True
                            st.success("✅ Thank you! You can continue chatting now.")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error("❌ Failed to save contact info. Please try again.")
                    else:
                        st.error("❌ Please fill all required fields (Name, Email, Phone)")
            
            st.stop()
        
        # Normal response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.ai.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.ai.save_to_db(prompt, response)

# Stunning Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="footer-container">
    <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0.5rem;'>
        <strong>🤖 Powered by Syngrid AI</strong>
    </p>
    <p style='color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 1rem;'>
        Built with 💜 using Streamlit, LangChain, HuggingFace & OpenRouter
    </p>
    <p style='color: rgba(255,255,255,0.6); font-size: 0.85rem;'>
        © 2025 Syngrid Technologies. All rights reserved.
    </p>
    <p style='color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;'>
        🌐 Visit us at <a href='https://syngrid.com' target='_blank' style='color: #667eea;'>syngrid.com</a>
    </p>
</div>
""", unsafe_allow_html=True)
