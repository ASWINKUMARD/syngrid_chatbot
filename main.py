import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
import hashlib
import time
import json
import base64
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

class FastScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.timeout = 6
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"@+/#]', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict:
        emails = set()
        phones = set()
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for email in re.findall(email_pattern, text):
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.css', '.js')):
                emails.add(email.lower())
        
        phone_patterns = [
            r'\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}',
            r'\d{4}[\s.-]\d{4}',
            r'\(\d{2,4}\)\s*\d{4}[\s.-]\d{4}',
        ]
        for pattern in phone_patterns:
            for phone in re.findall(pattern, text):
                cleaned = re.sub(r'[^\d+()]', '', phone)
                if 7 <= len(cleaned) <= 20:
                    phones.add(phone.strip())
        
        return {
            "emails": sorted(list(emails))[:5],
            "phones": sorted(list(phones))[:5]
        }
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                tag.decompose()
            
            title = ""
            if soup.find('title'):
                title = soup.find('title').get_text(strip=True)
            
            content = ""
            for selector in ['main', 'article', '[role="main"]', '.main-content', '#main', '.content']:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator='\n', strip=True)
                    if len(content) > 200:
                        break
            
            if len(content) < 200:
                texts = []
                for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
                content = '\n'.join(texts)
            
            lines = []
            seen = set()
            for line in content.split('\n'):
                line = self.clean_text(line)
                if len(line) > 25 and line.lower() not in seen:
                    lines.append(line)
                    seen.add(line.lower())
                if len(lines) >= 50:
                    break
            
            content = '\n'.join(lines)
            
            if len(content) < 100:
                return None
            
            return {
                "url": url,
                "title": title[:200],
                "content": content[:4000]
            }
            
        except Exception as e:
            return None
    
    def get_urls_to_scrape(self, base_url: str) -> List[str]:
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
        base_url = base_url.rstrip('/')
        
        paths = [
            '', '/about', '/about-us', '/services', '/products',
            '/contact', '/contact-us', '/pricing', '/solutions',
            '/home', '/index.html', '/company', '/team'
        ]
        
        urls = [f"{base_url}{path}" for path in paths]
        
        try:
            resp = requests.get(base_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                domain = urlparse(base_url).netloc
                
                for link in soup.find_all('a', href=True)[:60]:
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    if (urlparse(full_url).netloc == domain and 
                        not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.gif'])):
                        if full_url not in urls:
                            urls.append(full_url)
        except:
            pass
        
        return urls[:50]
    
    def scrape_website(self, base_url: str, progress_callback=None) -> Tuple[List[Dict], Dict]:
        start_time = time.time()
        
        urls = self.get_urls_to_scrape(base_url)
        
        pages = []
        all_text = ""
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self.scrape_page, url): url for url in urls}
            
            completed = 0
            for future in as_completed(future_to_url):
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(urls), future_to_url[future])
                
                try:
                    result = future.result()
                    if result:
                        pages.append(result)
                        all_text += "\n" + result['content']
                except:
                    pass
        
        contact_info = self.extract_contact_info(all_text)
        
        if len(pages) == 0:
            raise Exception(f"Could not scrape any content from {base_url}")
        
        return pages, contact_info

class SmartAI:
    def __init__(self):
        self.response_cache = {}
        
    def call_llm(self, prompt: str) -> str:
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not set."
        
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        for attempt in range(2):
            try:
                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 400,
                }
                
                resp = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=45)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content and len(content.strip()) > 5:
                            self.response_cache[cache_key] = content.strip()
                            return content.strip()
                
                if attempt < 1:
                    time.sleep(2)
                    
            except Exception as e:
                if attempt < 1:
                    time.sleep(2)
        
        return "I'm having trouble connecting right now. Please try again."

class UniversalChatbot:
    def __init__(self, company_name: str, website_url: str, slug: str = None):
        self.company_name = company_name
        self.website_url = website_url
        self.slug = slug or re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-')
        self.pages = []
        self.contact_info = {"emails": [], "phones": []}
        self.ready = False
        self.error = None
        self.ai = SmartAI()
        self.created_at = datetime.now().isoformat()
        
    def initialize(self, progress_callback=None):
        try:
            scraper = FastScraper()
            self.pages, self.contact_info = scraper.scrape_website(self.website_url, progress_callback)
            self.ready = True
            return True
        except Exception as e:
            self.error = str(e)
            return False
    
    def get_context(self, question: str) -> str:
        if not self.pages:
            return ""
        
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        scored_pages = []
        for page in self.pages:
            content_words = set(re.findall(r'\w+', page['content'].lower()))
            score = len(question_words & content_words)
            if score > 0:
                scored_pages.append((score, page))
        
        scored_pages.sort(reverse=True, key=lambda x: x[0])
        
        context_parts = []
        for score, page in scored_pages[:5]:
            context_parts.append(page['content'][:1000])
        
        return "\n\n---\n\n".join(context_parts)
    
    def ask(self, question: str) -> str:
        if not self.ready:
            return "⚠️ Chatbot not initialized yet."
        
        question_lower = question.lower().strip()
        
        greeting_words = ['hi', 'hello', 'hey', 'hai', 'good morning', 'good afternoon', 'good evening']
        if any(question_lower == g or question_lower.startswith(g + ' ') for g in greeting_words):
            return f"👋 Hello! I'm an AI assistant for **{self.company_name}**. How can I assist you?"
        
        contact_keywords = ['email', 'contact', 'phone', 'call', 'reach', 'address', 'location', 'office']
        if any(kw in question_lower for kw in contact_keywords):
            msg = f"📞 **Contact Information for {self.company_name}**\n\n"
            
            if self.contact_info['emails']:
                msg += "📧 **Email:**\n" + "\n".join([f"• {e}" for e in self.contact_info['emails']]) + "\n\n"
            
            if self.contact_info['phones']:
                msg += "📱 **Phone:**\n" + "\n".join([f"• {p}" for p in self.contact_info['phones']]) + "\n\n"
            
            if self.website_url:
                msg += f"🌐 **Website:** {self.website_url}"
            
            if not self.contact_info['emails'] and not self.contact_info['phones']:
                msg += f"Visit their website at {self.website_url} for contact details."
            
            return msg.strip()
        
        context = self.get_context(question)
        
        if not context or len(context) < 50:
            all_content = "\n".join([p['content'][:500] for p in self.pages[:3]])
            if all_content:
                context = all_content
            else:
                return f"I don't have specific information about that. Please visit {self.website_url}"
        
        prompt = f"""You are a helpful AI assistant for {self.company_name}.

Based on the following information from their website, answer the user's question clearly.

COMPANY INFORMATION:
{context[:2500]}

USER QUESTION: {question}

Instructions:
- Provide a helpful answer in 2-4 sentences
- Be specific and use details from the context
- Be friendly and professional

Answer:"""

        answer = self.ai.call_llm(prompt)
        return answer
    
    def to_dict(self):
        """Serialize chatbot for storage"""
        return {
            "company_name": self.company_name,
            "website_url": self.website_url,
            "slug": self.slug,
            "pages": self.pages,
            "contact_info": self.contact_info,
            "ready": self.ready,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize chatbot from storage"""
        bot = cls(data["company_name"], data["website_url"], data["slug"])
        bot.pages = data["pages"]
        bot.contact_info = data["contact_info"]
        bot.ready = data["ready"]
        bot.created_at = data.get("created_at", datetime.now().isoformat())
        return bot

def generate_shareable_link(slug: str) -> str:
    """Generate shareable public link for chatbot"""
    base_url = "https://your-app-domain.streamlit.app"  # Replace with your actual Streamlit URL
    return f"{base_url}?bot={slug}"

def generate_embed_code(slug: str, company_name: str) -> str:
    """Generate embed code for external websites"""
    base_url = "https://your-app-domain.streamlit.app"  # Replace with your actual URL
    
    embed_code = f'''<!-- {company_name} AI Chatbot -->
<div id="ai-chatbot-{slug}"></div>
<script>
  (function() {{
    var iframe = document.createElement('iframe');
    iframe.src = '{base_url}?bot={slug}&embed=true';
    iframe.style.width = '400px';
    iframe.style.height = '600px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '10px';
    iframe.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
    document.getElementById('ai-chatbot-{slug}').appendChild(iframe);
  }})();
</script>'''
    
    return embed_code

def save_chatbots():
    """Save all chatbots to session state (could be extended to database)"""
    if 'chatbots' in st.session_state:
        chatbots_data = {}
        for slug, bot in st.session_state.chatbots.items():
            chatbots_data[slug] = bot.to_dict()
        st.session_state.chatbots_data = chatbots_data

def load_chatbots():
    """Load chatbots from session state"""
    if 'chatbots_data' in st.session_state:
        st.session_state.chatbots = {}
        for slug, data in st.session_state.chatbots_data.items():
            st.session_state.chatbots[slug] = UniversalChatbot.from_dict(data)

def init_session():
    if 'chatbots' not in st.session_state:
        st.session_state.chatbots = {}
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    load_chatbots()

def main():
    st.set_page_config(
        page_title="Universal AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session()
    
    # Check for URL parameters (for shareable links)
    query_params = st.query_params
    bot_slug = query_params.get("bot", None)
    is_embed = query_params.get("embed", "false") == "true"
    
    # If accessing via shareable link and bot exists
    if bot_slug and bot_slug in st.session_state.chatbots:
        st.session_state.current_company = bot_slug
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    # Hide sidebar for embed mode
    if is_embed:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        .main { padding: 1rem; }
        </style>
        """, unsafe_allow_html=True)
    
    if not is_embed:
        st.title("🤖 Universal AI Chatbot")
        st.caption("Create shareable AI chatbots for ANY company!")
    
    # API Key Check
    if not OPENROUTER_API_KEY and not is_embed:
        st.error("⚠️ OPENROUTER_API_KEY not set!")
        st.info("Set your API key: `export OPENROUTER_API_KEY='your_key'`")
        st.info("Get free key: https://openrouter.ai/keys")
        return
    
    # Sidebar - Company Management (hidden in embed mode)
    if not is_embed:
        st.sidebar.title("🏢 Company Management")
        
        with st.sidebar.expander("➕ Add New Company", expanded=True):
            company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
            website_url = st.text_input("Website URL", placeholder="https://example.com")
            
            if st.button("🚀 Create Chatbot", type="primary"):
                if not company_name or not website_url:
                    st.warning("Please provide both name and URL.")
                else:
                    slug = re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-')
                    
                    with st.spinner(f"Creating chatbot for {company_name}..."):
                        progress = st.progress(0)
                        status = st.empty()
                        
                        def callback(done, total, url):
                            progress.progress(done / max(total, 1))
                            status.text(f"Scraping {done}/{total}...")
                        
                        chatbot = UniversalChatbot(company_name, website_url, slug)
                        success = chatbot.initialize(callback)
                        
                        if success:
                            st.session_state.chatbots[slug] = chatbot
                            st.session_state.current_company = slug
                            st.session_state.chat_history = []
                            save_chatbots()
                            st.success(f"✅ Chatbot ready!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed: {chatbot.error}")
        
        # Show existing chatbots
        if st.session_state.chatbots:
            st.sidebar.subheader("📋 Your Chatbots")
            
            for slug, bot in st.session_state.chatbots.items():
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    if st.button(f"💬 {bot.company_name}", key=f"select_{slug}"):
                        st.session_state.current_company = slug
                        st.session_state.chat_history = []
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{slug}"):
                        del st.session_state.chatbots[slug]
                        if st.session_state.current_company == slug:
                            st.session_state.current_company = None
                        save_chatbots()
                        st.rerun()
    
    # Main Chat Interface
    if not st.session_state.current_company:
        if not is_embed:
            st.info("👈 Create a new chatbot to get started!")
            st.markdown("### 🎯 Features:")
            st.markdown("""
            - ✨ **Instant Setup** - Just provide company name and website
            - 🔗 **Shareable Links** - Get public URL for each chatbot
            - 📦 **Embed Code** - Add chatbot to any website
            - 🚀 **Fast Scraping** - Intelligent parallel scraping
            - 🧠 **Smart AI** - Context-aware responses
            """)
    else:
        chatbot = st.session_state.chatbots[st.session_state.current_company]
        
        # Header
        if not is_embed:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"💬 {chatbot.company_name}")
            with col2:
                if st.button("🔗 Share"):
                    st.session_state.show_share = True
            with col3:
                if st.button("🔄 Refresh"):
                    with st.spinner("Refreshing..."):
                        chatbot.initialize()
                        save_chatbots()
                        st.success("✅ Refreshed!")
                        st.rerun()
            
            # Share Modal
            if st.session_state.get('show_share', False):
                with st.expander("🔗 Share This Chatbot", expanded=True):
                    shareable_link = generate_shareable_link(chatbot.slug)
                    embed_code = generate_embed_code(chatbot.slug, chatbot.company_name)
                    
                    st.markdown("**📎 Public Shareable Link:**")
                    st.code(shareable_link, language="text")
                    if st.button("📋 Copy Link"):
                        st.success("✅ Copy the link above!")
                    
                    st.markdown("**🔧 Embed Code (Add to your website):**")
                    st.code(embed_code, language="html")
                    if st.button("📋 Copy Embed Code"):
                        st.success("✅ Copy the embed code above!")
                    
                    if st.button("❌ Close"):
                        st.session_state.show_share = False
                        st.rerun()
            
            # Info panel
            with st.expander("ℹ️ Chatbot Info"):
                st.write(f"**Website:** {chatbot.website_url}")
                st.write(f"**Pages Scraped:** {len(chatbot.pages)}")
                st.write(f"**Emails Found:** {len(chatbot.contact_info['emails'])}")
                st.write(f"**Phones Found:** {len(chatbot.contact_info['phones'])}")
                st.write(f"**Created:** {chatbot.created_at[:10]}")
        else:
            st.markdown(f"### 💬 {chatbot.company_name}")
        
        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if user_input := st.chat_input("Ask anything..."):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chatbot.ask(user_input)
                st.markdown(answer)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
