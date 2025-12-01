import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from collections import deque
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import re
import os

# Page config
st.set_page_config(
    page_title="Syngrid AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .contact-info {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Database setup
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

# Validation functions
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    return len(cleaned) == 10 and cleaned.isdigit()

def validate_name(name):
    return len(name.strip()) >= 2 and re.match(r'^[a-zA-Z\s]+$', name.strip()) is not None

def save_user_contact(name, email, phone):
    try:
        db = SessionLocal()
        contact = UserContact(name=name, email=email, phone=phone, timestamp=datetime.now(timezone.utc))
        db.add(contact)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
SYNGRID_WEBSITE = "https://syngrid.com/"
MODEL = "kwaipilot/kat-coder-pro:free"

PRIORITY_PAGES = [
    "", "about", "about-us", "services", "solutions", "products", "contact",
    "contact-us", "team", "careers", "blog", "case-studies", "portfolio",
    "industries", "technology", "expertise", "what-we-do", "who-we-are",
    "footer", "locations", "reach-us"
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
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif')):
                self.company_info['emails'].add(email.lower())
        
        # Extract phone numbers
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
        
        # Extract addresses from contact pages
        if any(keyword in url.lower() for keyword in ['contact', 'footer', 'about', 'reach']):
            lines = text.split('\n')
            
            # India address
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
                                if not any(skip in current_line.lower() for skip in ['career', 'job', 'hiring', 'vacancy', 'apply', 'opening', 'position']):
                                    address_parts.append(current_line)
                        
                        if address_parts:
                            full_address = ' '.join(address_parts)
                            if 'madurai' in full_address.lower() and len(full_address) > 30:
                                address_clean = self.clean_address(full_address)
                                if len(address_clean) > 20 and len(address_clean) < 300:
                                    self.company_info['india_address'] = address_clean
                                    break
            
            # Singapore address
            if not self.company_info['singapore_address']:
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if 'singapore' in line_lower and (re.search(r'\d{6}', line) or 'road' in line_lower or 'street' in line_lower):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        
                        address_parts = []
                        for j in range(start, end):
                            current_line = lines[j].strip()
                            if current_line and len(current_line) > 5:
                                if not any(skip in current_line.lower() for skip in ['career', 'job', 'hiring', 'vacancy', 'apply', 'opening', 'position']):
                                    address_parts.append(current_line)
                        
                        if address_parts:
                            full_address = ' '.join(address_parts)
                            if 'singapore' in full_address.lower() and len(full_address) > 30:
                                address_clean = self.clean_address(full_address)
                                if len(address_clean) > 20 and len(address_clean) < 300:
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
        content_dict = {
            'url': url,
            'title': '',
            'main_content': '',
            'metadata': {}
        }
        
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

    def scrape_website(self, base_url, max_pages=50, progress_callback=None):
        visited = set()
        all_content = []
        queue = deque()
        base_domain = urlparse(base_url).netloc
        
        for page in PRIORITY_PAGES:
            priority_url = urljoin(base_url, page)
            queue.append(priority_url)
        
        if base_url not in queue:
            queue.append(base_url)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

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
                    formatted_content = f"""URL: {content_data['url']}
TITLE: {content_data['title']}
"""
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
                        
                        if (next_url not in visited and 
                            next_url not in queue and 
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
            info_text += "INDIA OFFICE (Corporate Office):\n"
            info_text += f"  {self.company_info['india_address']}\n\n"
        
        if self.company_info['singapore_address']:
            info_text += "SINGAPORE OFFICE:\n"
            info_text += f"  {self.company_info['singapore_address']}\n\n"
        
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
                response += "📍 **India Office (Corporate Office):**\n"
                response += f"   {self.company_info['india_address']}\n\n"
            
            if self.company_info['singapore_address']:
                response += "📍 **Singapore Office:**\n"
                response += f"   {self.company_info['singapore_address']}\n\n"
        
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

    def initialize(self, url, max_pages=50, progress_callback=None):
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

            vectorstore = FAISS.from_texts(
                chunks,
                embedding=embeddings
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
            return "⚠️ Please initialize the assistant first."

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
                    {
                        "role": "system",
                        "content": "You are Syngrid AI Assistant. Answer questions about Syngrid Technologies based only on the given context. Be concise and helpful."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }

            response = requests.post(
                OPENROUTER_API_BASE,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
                if answer and len(answer) > 10:
                    self.cache[q_lower] = answer
                    return answer
                else:
                    return "I'm having trouble generating a response. Please try again."
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                return f"⚠️ API Error (Status {response.status_code}): {error_msg}"

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
            st.error(f"Database error: {str(e)}")


# Initialize session state
if 'ai' not in st.session_state:
    st.session_state.ai = SyngridAI()
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.question_count = 0
    st.session_state.user_info_collected = False

# Main UI
st.markdown('<div class="main-header"><h1>⚡ Syngrid AI Assistant</h1><p>Powered by AI - Ask me anything about Syngrid Technologies</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Settings")
    
    if not st.session_state.initialized:
        st.info("👋 Welcome! Click below to initialize the assistant.")
        
        if st.button("🚀 Initialize Assistant"):
            with st.spinner("🔄 Initializing... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total, url):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Scraping: {current}/{total} pages")
                
                success = st.session_state.ai.initialize(
                    SYNGRID_WEBSITE, 
                    max_pages=50,
                    progress_callback=progress_callback
                )
                
                if success:
                    st.session_state.initialized = True
                    st.success("✅ Assistant initialized successfully!")
                    st.rerun()
                else:
                    st.error("❌ Initialization failed. Please try again.")
    else:
        st.success("✅ Assistant is ready!")
        st.metric("Pages Scraped", st.session_state.ai.status["pages_scraped"])
        st.metric("Messages", len(st.session_state.messages))
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.session_state.question_count = 0
            st.session_state.ai.cache.clear()
            st.rerun()
        
        if st.button("📞 Show Contact Info"):
            contact_info = st.session_state.ai.get_company_contact_info()
            st.markdown(f'<div class="contact-info">{contact_info}</div>', unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 💡 Quick Commands")
    st.markdown("""
    - Ask about services
    - Request contact info
    - Learn about products
    - Explore solutions
    """)

# Main chat interface
if st.session_state.initialized:
    # Display chat messages
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        icon = "👤" if message["role"] == "user" else "🤖"
        st.markdown(f'<div class="chat-message {role_class}"><strong>{icon} {message["role"].title()}:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    # User input form (after checking for contact info collection)
    if st.session_state.question_count == 3 and not st.session_state.user_info_collected:
        st.markdown("---")
        st.markdown("### 📝 Contact Information Required")
        st.info("To continue using the assistant, please provide your details:")
        
        with st.form("contact_form"):
            name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email Address", placeholder="john@example.com")
            phone = st.text_input("Phone Number (10 digits)", placeholder="9876543210")
            
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if not validate_name(name):
                    st.error("❌ Invalid name. Please enter at least 2 characters (letters only).")
                elif not validate_email(email):
                    st.error("❌ Invalid email. Please enter a valid email address.")
                elif not validate_phone(phone):
                    st.error("❌ Invalid phone number. Please enter 10 digits.")
                else:
                    cleaned_phone = re.sub(r'[\s\-\(\)]', '', phone)
                    if save_user_contact(name, email, cleaned_phone):
                        st.session_state.user_info_collected = True
                        st.success("✅ Thank you! Your information has been saved. You can continue asking questions.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save contact information. Please try again.")
    else:
        # Chat input
        user_input = st.chat_input("💬 Ask me anything about Syngrid...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.question_count += 1
            
            # Get AI response
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.ai.ask(user_input)
                st.session_state.ai.save_to_db(user_input, response)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()
else:
    st.info("👈 Please initialize the assistant from the sidebar to start chatting.")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">Built with ❤️ using Streamlit | © 2024 Syngrid Technologies</div>',
    unsafe_allow_html=True
)
