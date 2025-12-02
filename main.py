import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from collections import deque
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import re
import os
import time

# Page Configuration
st.set_page_config(
    page_title="Syngrid AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .stChatFloatingInputContainer {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Custom header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-loading {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Contact form */
    .contact-form {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
SYNGRID_WEBSITE = "https://syngrid.com/"
MODEL = "kwaipilot/kat-coder-pro:free"

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
                    return "I'm having trouble generating a response. Please try again."
            else:
                return f"⚠️ API Error (Status {response.status_code})"

        except Exception as e:
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
        st.markdown('<div class="status-badge status-ready">✅ Ready to Chat</div>', unsafe_allow_html=True)
        st.metric("Pages Scraped", st.session_state.ai.status["pages_scraped"])
        st.metric("Emails Found", len(st.session_state.ai.company_info['emails']))
        st.metric("Conversations", len(st.session_state.messages))
    else:
        st.markdown('<div class="status-badge status-loading pulse">🔄 Initializing...</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Quick Actions")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()
    
    if st.button("📞 Show Contact Info", use_container_width=True):
        if st.session_state.initialized:
            contact_info = st.session_state.ai.get_company_contact_info()
            st.info(contact_info)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    - ✅ Real-time AI responses
    - ✅ 40+ pages scraped
    - ✅ Contact information
    - ✅ Smart caching
    - ✅ Conversation history
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Syngrid AI Assistant uses advanced NLP to answer your questions about Syngrid Technologies instantly.")

# Auto-initialize on first load
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing Syngrid AI Assistant..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, url):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Scraping page {current}/{total}: {url[:50]}...")
        
        success = st.session_state.ai.initialize(SYNGRID_WEBSITE, max_pages=40, progress_callback=update_progress)
        
        if success:
            st.session_state.initialized = True
            progress_bar.progress(1.0)
            status_text.text("✅ Initialization complete!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Failed to initialize. Please refresh the page.")
            st.stop()

# Chat Interface
if st.session_state.initialized:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Syngrid Technologies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.question_count += 1
        
        # Show contact form after 3 questions
        if st.session_state.question_count == 3 and not st.session_state.user_info_collected:
            with st.chat_message("assistant"):
                response = st.session_state.ai.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.ai.save_to_db(prompt, response)
            
            st.markdown("---")
            st.markdown('<div class="contact-form">', unsafe_allow_html=True)
            st.markdown("### 📋 Please provide your contact information to continue")
            
            with st.form("contact_form"):
                name = st.text_input("Full Name *", placeholder="John Doe")
                email = st.text_input("Email Address *", placeholder="john@example.com")
                phone = st.text_input("Phone Number *", placeholder="+91 98765 43210")
                
                submitted = st.form_submit_button("Submit & Continue", use_container_width=True)
                
                if submitted:
                    if name and email and phone:
                        try:
                            db = SessionLocal()
                            contact = UserContact(name=name, email=email, phone=phone, timestamp=datetime.now(timezone.utc))
                            db.add(contact)
                            db.commit()
                            db.close()
                            st.session_state.user_info_collected = True
                            st.success("✅ Thank you! You can continue chatting.")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error("Failed to save contact info. Please try again.")
                    else:
                        st.error("Please fill all fields.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()
        
        # Normal response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.ai.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.ai.save_to_db(prompt, response)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.7); padding: 2rem;'>
    <p>🤖 Powered by Syngrid AI | Built with Streamlit & OpenRouter</p>
    <p>© 2025 Syngrid Technologies. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
