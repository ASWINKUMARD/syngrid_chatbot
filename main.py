import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
import hashlib
import time
import json
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[ENV] ✅ Environment variables loaded from .env file")
except ImportError:
    print("[ENV] ⚠️ python-dotenv not installed. Using system environment variables.")

# ========================================
# CONFIGURATION
# ========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

# Get the deployment URL from environment or use default
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL", "http://localhost:8501")

print("\n" + "="*50)
print("CONFIGURATION CHECK")
print("="*50)
print(f"OPENROUTER_API_KEY: {'SET ✅' if OPENROUTER_API_KEY else 'NOT SET ❌'}")
print(f"DEPLOYMENT_URL: {DEPLOYMENT_URL}")
print("="*50 + "\n")


# ========================================
# IN-MEMORY DATA STORAGE
# ========================================
class InMemoryStorage:
    """Handles all data storage in memory"""
   
    def __init__(self):
        self.leads = []
        self.chatbots = {}
        self.next_lead_id = 1
        self.next_chatbot_id = 1
   
    def save_lead(self, chatbot_id, company_name, user_name, user_email,
                  user_phone, session_id, questions_asked, conversation):
        """Save lead to in-memory storage"""
        try:
            lead = {
                'userid': self.next_lead_id,
                'username': user_name or "Anonymous",
                'mailid': user_email or "not_provided@example.com",
                'phonenumber': user_phone or "Not provided",
                'conversation': json.dumps(conversation) if conversation else "[]",
                'timestart': datetime.now(),
                'timeend': None,
                'chatbot_id': chatbot_id,
                'company_name': company_name,
                'session_id': session_id,
                'questions_asked': questions_asked
            }
           
            self.leads.append(lead)
            self.next_lead_id += 1
            print(f"[Storage] ✅ Lead saved successfully with ID: {lead['userid']}")
            return True
        except Exception as e:
            print(f"[Storage] ❌ Save lead error: {e}")
            st.error(f"Failed to save lead: {e}")
            return False
   
    def get_leads(self, chatbot_id=None):
        """Retrieve leads from in-memory storage"""
        try:
            if chatbot_id:
                return [lead for lead in self.leads if lead['chatbot_id'] == chatbot_id]
            return self.leads
        except Exception as e:
            print(f"[Storage] ❌ Get leads error: {e}")
            return []
   
    def save_chatbot(self, chatbot_id, company_name, website_url, embed_code):
        """Save or update chatbot configuration"""
        try:
            self.chatbots[chatbot_id] = {
                'id': self.next_chatbot_id,
                'chatbot_id': chatbot_id,
                'company_name': company_name,
                'website_url': website_url,
                'embed_code': embed_code,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            self.next_chatbot_id += 1
            print(f"[Storage] ✅ Chatbot saved: {company_name}")
            return True
        except Exception as e:
            print(f"[Storage] ❌ Chatbot save error: {e}")
            return False
   
    def get_chatbot(self, chatbot_id):
        """Get chatbot configuration"""
        return self.chatbots.get(chatbot_id)
    
    def get_chatbot_by_id(self, chatbot_id):
        """Get chatbot by ID for public access"""
        for bot_data in self.chatbots.values():
            if bot_data['chatbot_id'] == chatbot_id:
                return bot_data
        return None


# Initialize global storage instance
storage = InMemoryStorage()


# ========================================
# WEBSITE SCRAPER
# ========================================
class FastScraper:
    """Fast website scraper with multi-threading"""
   
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        self.timeout = 6
   
    def scrape_page(self, url):
        """Scrape a single page"""
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return None
           
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
           
            content = soup.get_text(separator='\n', strip=True)
            lines = [l.strip() for l in content.split('\n') if len(l.strip()) > 25][:50]
           
            return {"url": url, "content": '\n'.join(lines)[:4000]} if lines else None
        except Exception as e:
            print(f"[Scraper] Error scraping {url}: {e}")
            return None
   
    def scrape_website(self, base_url, progress_callback=None):
        """Scrape multiple pages from a website"""
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
       
        urls = [base_url, f"{base_url}/about", f"{base_url}/services",
                f"{base_url}/contact", f"{base_url}/products"]
       
        pages = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.scrape_page, url): url for url in urls}
            for i, future in enumerate(as_completed(futures)):
                if progress_callback:
                    progress_callback(i+1, len(urls), futures[future])
                result = future.result()
                if result:
                    pages.append(result)
       
        all_text = '\n'.join([p['content'] for p in pages])
        emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', all_text)))[:3]
        phones = list(set(re.findall(r'\+?\d[\d\s.-]{7,}\d', all_text)))[:3]
       
        return pages, {"emails": emails, "phones": phones}


# ========================================
# AI INTEGRATION
# ========================================
class SmartAI:
    """AI integration with caching"""
   
    def __init__(self):
        self.cache = {}
   
    def call_llm(self, prompt):
        """Call LLM API with caching"""
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not set. Please configure OPENROUTER_API_KEY."
       
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:12]
        if cache_key in self.cache:
            print("[AI] Cache hit")
            return self.cache[cache_key]
       
        try:
            resp = requests.post(
                OPENROUTER_API_BASE,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": DEPLOYMENT_URL,
                    "X-Title": "AI Chatbot Lead Generator"
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400
                },
                timeout=30
            )
           
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data:
                    answer = data["choices"][0]["message"]["content"].strip()
                    self.cache[cache_key] = answer
                    return answer
           
            print(f"[AI] API Response Status: {resp.status_code}")
            print(f"[AI] API Response: {resp.text}")
           
            if resp.status_code == 401:
                return "⚠️ API Authentication Failed. Please check your OPENROUTER_API_KEY."
            elif resp.status_code == 402:
                return "⚠️ Insufficient credits. Please add credits to your OpenRouter account."
            elif resp.status_code == 429:
                return "⚠️ Rate limit exceeded. Please try again in a moment."
            else:
                return f"⚠️ API Error {resp.status_code}: {resp.text[:100]}"
               
        except Exception as e:
            print(f"[AI] Error: {e}")
            return "I'm having connection issues. Please try again."


# ========================================
# CHATBOT CLASS
# ========================================
class UniversalChatbot:
    """Universal chatbot that works for any website"""
   
    def __init__(self, company_name, website_url, chatbot_id):
        self.company_name = company_name
        self.website_url = website_url
        self.chatbot_id = chatbot_id
        self.pages = []
        self.contact_info = {}
        self.ready = False
        self.ai = SmartAI()
   
    def initialize(self, progress_callback=None):
        """Initialize chatbot by scraping website"""
        try:
            scraper = FastScraper()
            self.pages, self.contact_info = scraper.scrape_website(self.website_url, progress_callback)
            self.ready = True
            print(f"[Bot] Initialized for {self.company_name}")
            return True
        except Exception as e:
            print(f"[Bot] Initialization error: {e}")
            return False
   
    def ask(self, question):
        """Process user question and generate response"""
        if not self.ready:
            return "⚠️ Chatbot not ready. Please try again."
       
        if any(g in question.lower() for g in ['hi', 'hello', 'hey']):
            return f"👋 Hello! I'm the AI assistant for **{self.company_name}**. How can I help you today?"
       
        if any(k in question.lower() for k in ['email', 'contact', 'phone']):
            msg = f"📞 **Contact {self.company_name}**\n\n"
            if self.contact_info.get('emails'):
                msg += "📧 " + ", ".join(self.contact_info['emails']) + "\n"
            if self.contact_info.get('phones'):
                msg += "📱 " + ", ".join(self.contact_info['phones']) + "\n"
            msg += f"🌐 {self.website_url}"
            return msg
       
        context = '\n'.join([p['content'][:800] for p in self.pages[:3]])
       
        prompt = f"""You are a helpful assistant for {self.company_name}.

Context from their website:
{context}

User question: {question}

Provide a helpful, natural 2-3 sentence answer.

Answer:"""
       
        return self.ai.call_llm(prompt)


# ========================================
# EMBED CODE GENERATION
# ========================================
def generate_embed_code(chatbot_id, company_name):
    """Generate production-ready HTML embed code"""
    embed_url = f"{DEPLOYMENT_URL}?mode=embed&id={chatbot_id}"
    
    return f'''<!-- {company_name} AI Chatbot Widget -->
<div id="ai-chatbot-{chatbot_id}"></div>
<script>
(function() {{
  // Configuration
  var chatbotId = '{chatbot_id}';
  var embedUrl = '{embed_url}';
  
  // Create chat button
  var chatButton = document.createElement('button');
  chatButton.id = 'ai-chat-btn-' + chatbotId;
  chatButton.innerHTML = '💬 Chat with us';
  chatButton.style.cssText = 'position:fixed;bottom:20px;right:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;border-radius:50px;padding:15px 30px;font-size:16px;font-weight:600;cursor:pointer;box-shadow:0 4px 15px rgba(102,126,234,0.4);z-index:999999;transition:all 0.3s ease;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;';
  
  // Create iframe container
  var chatContainer = document.createElement('div');
  chatContainer.id = 'ai-chat-container-' + chatbotId;
  chatContainer.style.cssText = 'position:fixed;bottom:90px;right:20px;width:400px;height:600px;max-width:calc(100vw - 40px);max-height:calc(100vh - 120px);border:none;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,0.3);z-index:999998;display:none;overflow:hidden;background:white;';
  
  // Create iframe
  var chatIframe = document.createElement('iframe');
  chatIframe.src = embedUrl;
  chatIframe.style.cssText = 'width:100%;height:100%;border:none;border-radius:12px;';
  chatIframe.allow = 'clipboard-write';
  
  // Append iframe to container
  chatContainer.appendChild(chatIframe);
  
  // Toggle chat visibility
  chatButton.onclick = function() {{
    var isVisible = chatContainer.style.display === 'block';
    chatContainer.style.display = isVisible ? 'none' : 'block';
    chatButton.innerHTML = isVisible ? '💬 Chat with us' : '✕ Close';
    chatButton.style.background = isVisible ? 
      'linear-gradient(135deg,#667eea 0%,#764ba2 100%)' : 
      'linear-gradient(135deg,#f093fb 0%,#f5576c 100%)';
  }};
  
  // Hover effect
  chatButton.onmouseenter = function() {{
    this.style.transform = 'scale(1.05)';
    this.style.boxShadow = '0 6px 20px rgba(102,126,234,0.5)';
  }};
  chatButton.onmouseleave = function() {{
    this.style.transform = 'scale(1)';
    this.style.boxShadow = '0 4px 15px rgba(102,126,234,0.4)';
  }};
  
  // Append to body when DOM is ready
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', function() {{
      document.body.appendChild(chatButton);
      document.body.appendChild(chatContainer);
    }});
  }} else {{
    document.body.appendChild(chatButton);
    document.body.appendChild(chatContainer);
  }}
  
  // Mobile responsiveness
  if (window.innerWidth < 480) {{
    chatContainer.style.width = 'calc(100vw - 20px)';
    chatContainer.style.height = 'calc(100vh - 100px)';
    chatContainer.style.right = '10px';
    chatContainer.style.bottom = '80px';
    chatButton.style.right = '10px';
  }}
}})();
</script>'''


def generate_wordpress_embed(chatbot_id, company_name):
    """Generate WordPress-specific embed code"""
    return f'''<!-- {company_name} AI Chatbot for WordPress -->
<!-- Add this to your theme's footer.php or use a Custom HTML widget -->

{generate_embed_code(chatbot_id, company_name)}

<!-- Alternative: Add via WordPress Customizer -->
<!-- Go to Appearance > Customize > Additional CSS and paste the widget code -->
'''


def generate_shopify_embed(chatbot_id, company_name):
    """Generate Shopify-specific embed code"""
    return f'''<!-- {company_name} AI Chatbot for Shopify -->
<!-- Add this to your theme.liquid file before </body> tag -->
<!-- Or go to Online Store > Themes > Edit Code > theme.liquid -->

{generate_embed_code(chatbot_id, company_name)}
'''


def generate_wix_embed(chatbot_id, company_name):
    """Generate Wix-specific embed instructions"""
    return f'''<!-- {company_name} AI Chatbot for Wix -->

INSTRUCTIONS FOR WIX:
1. Go to your Wix Editor
2. Click on the "+" button to add elements
3. Select "Embed" > "Custom Embeds" > "Embed a Widget"
4. Paste the code below into the "Code" section
5. Click "Update" and position the widget

CODE TO PASTE:
{generate_embed_code(chatbot_id, company_name)}
'''


# ========================================
# UTILITY FUNCTIONS
# ========================================
def validate_email(email):
    """Validate email format"""
    if not email or not email.strip():
        return False
    return '@' in email and '.' in email.split('@')[-1]


def init_session():
    """Initialize session state variables"""
    defaults = {
        'chatbots': {},
        'current_company': None,
        'chat_history': [],
        'question_count': 0,
        'lead_capture_mode': None,
        'lead_data': {},
        'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16],
        'lead_captured': False,
        'embed_mode': False,
        'embed_chatbot_id': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ========================================
# EMBED MODE INTERFACE
# ========================================
def render_embed_mode():
    """Render chatbot in embed mode for external websites"""
    st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")
    
    # Custom CSS for embed mode
    st.markdown("""
    <style>
        .stApp {
            background: transparent;
        }
        .main .block-container {
            padding: 1rem;
            max-width: 100%;
        }
        header, footer {
            display: none !important;
        }
        .stChatFloatingInputContainer {
            bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    chatbot_id = st.query_params.get("id")
    
    if not chatbot_id:
        st.error("⚠️ No chatbot ID provided")
        return
    
    # Get chatbot data
    bot_data = storage.get_chatbot_by_id(chatbot_id)
    
    if not bot_data:
        st.error("⚠️ Chatbot not found. Please create it first in the admin panel.")
        st.info(f"Looking for chatbot ID: {chatbot_id}")
        return
    
    # Initialize chatbot if not in session
    if chatbot_id not in st.session_state.chatbots:
        bot = UniversalChatbot(
            bot_data['company_name'],
            bot_data['website_url'],
            chatbot_id
        )
        
        with st.spinner("Initializing chatbot..."):
            if bot.initialize():
                st.session_state.chatbots[chatbot_id] = bot
            else:
                st.error("Failed to initialize chatbot")
                return
    
    bot = st.session_state.chatbots[chatbot_id]
    
    # Initialize session for this embed
    if 'embed_chat_history' not in st.session_state:
        st.session_state.embed_chat_history = []
    if 'embed_question_count' not in st.session_state:
        st.session_state.embed_question_count = 0
    if 'embed_lead_captured' not in st.session_state:
        st.session_state.embed_lead_captured = False
    if 'embed_lead_mode' not in st.session_state:
        st.session_state.embed_lead_mode = None
    if 'embed_lead_data' not in st.session_state:
        st.session_state.embed_lead_data = {}
    
    # Header
    st.markdown(f"### 💬 {bot.company_name}")
    st.caption("AI-powered assistant")
    
    # Display chat history
    for msg in st.session_state.embed_chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Lead capture form (similar to main app)
    if st.session_state.embed_lead_mode and not st.session_state.embed_lead_captured:
        st.markdown("---")
        st.markdown("### 🎯 Let us help you better!")
        
        if st.session_state.embed_lead_mode == 'ask_name':
            name = st.text_input("Your Name", key="embed_name", placeholder="John Doe")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Submit", key="embed_submit_name"):
                    if name and name.strip():
                        st.session_state.embed_lead_data['name'] = name.strip()
                        st.session_state.embed_lead_mode = 'ask_email'
                        st.rerun()
            with col2:
                if st.button("⏭️ Skip", key="embed_skip_name"):
                    st.session_state.embed_lead_data['name'] = "Anonymous"
                    st.session_state.embed_lead_mode = 'ask_email'
                    st.rerun()
        
        elif st.session_state.embed_lead_mode == 'ask_email':
            email = st.text_input("Your Email", key="embed_email", placeholder="john@example.com")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Submit", key="embed_submit_email"):
                    if email and validate_email(email):
                        st.session_state.embed_lead_data['email'] = email.strip()
                        st.session_state.embed_lead_mode = 'ask_phone'
                        st.rerun()
            with col2:
                if st.button("⏭️ Skip", key="embed_skip_email"):
                    st.session_state.embed_lead_data['email'] = "not_provided@example.com"
                    st.session_state.embed_lead_mode = 'ask_phone'
                    st.rerun()
        
        elif st.session_state.embed_lead_mode == 'ask_phone':
            phone = st.text_input("Your Phone", key="embed_phone", placeholder="+1234567890")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Submit", key="embed_submit_phone"):
                    phone_value = phone.strip() if phone else "Not provided"
                    st.session_state.embed_lead_data['phone'] = phone_value
                    
                    storage.save_lead(
                        chatbot_id,
                        bot.company_name,
                        st.session_state.embed_lead_data.get('name', 'Anonymous'),
                        st.session_state.embed_lead_data.get('email', 'not_provided@example.com'),
                        st.session_state.embed_lead_data.get('phone', 'Not provided'),
                        st.session_state.session_id,
                        st.session_state.embed_question_count,
                        st.session_state.embed_chat_history
                    )
                    
                    st.session_state.embed_lead_captured = True
                    st.session_state.embed_lead_mode = None
                    st.success("✅ Thank you!")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("⏭️ Skip", key="embed_skip_phone"):
                    storage.save_lead(
                        chatbot_id,
                        bot.company_name,
                        st.session_state.embed_lead_data.get('name', 'Anonymous'),
                        st.session_state.embed_lead_data.get('email', 'not_provided@example.com'),
                        "Not provided",
                        st.session_state.session_id,
                        st.session_state.embed_question_count,
                        st.session_state.embed_chat_history
                    )
                    
                    st.session_state.embed_lead_captured = True
                    st.session_state.embed_lead_mode = None
                    st.success("✅ Thank you!")
                    time.sleep(1)
                    st.rerun()
    
    # Chat input
    if question := st.chat_input("Ask anything...", 
                                  disabled=bool(st.session_state.embed_lead_mode)):
        st.session_state.embed_chat_history.append({"role": "user", "content": question})
        
        with st.spinner("Thinking..."):
            answer = bot.ask(question)
        
        st.session_state.embed_chat_history.append({"role": "assistant", "content": answer})
        st.session_state.embed_question_count += 1
        
        # Trigger lead capture after 3 questions
        if st.session_state.embed_question_count >= 3 and \
           not st.session_state.embed_lead_captured and \
           not st.session_state.embed_lead_mode:
            st.session_state.embed_lead_mode = 'ask_name'
        
        st.rerun()


# ========================================
# MAIN APPLICATION
# ========================================
def main():
    """Main Streamlit application"""
    
    # Check if running in embed mode
    mode = st.query_params.get("mode")
    if mode == "embed":
        render_embed_mode()
        return
    
    # Regular admin interface
    st.set_page_config(page_title="AI Chatbot Lead Generator", page_icon="🤖", layout="wide")
    init_session()
   
    st.title("🤖 Universal AI Chatbot with Lead Capture")
    st.caption("Create chatbots • Capture leads • Deploy anywhere")
   
    # API Key Check
    if not OPENROUTER_API_KEY:
        st.error("⚠️ OPENROUTER_API_KEY not set!")
        st.info("Set environment variable: OPENROUTER_API_KEY='your_key'")
        
        with st.expander("🔑 How to get an API Key"):
            st.markdown("""
            1. Visit [OpenRouter.ai](https://openrouter.ai/)
            2. Sign up/Login
            3. Go to [Keys](https://openrouter.ai/keys)
            4. Create API key
            5. Add credits (Settings → Credits)
            6. Set as environment variable
            """)
        st.stop()
   
    # Sidebar - Management
    st.sidebar.title("🏢 Management Panel")
   
    # API Status
    with st.sidebar.expander("🔑 API Status"):
        key_preview = f"{OPENROUTER_API_KEY[:8]}...{OPENROUTER_API_KEY[-4:]}"
        st.success(f"✅ Key: {key_preview}")
        st.info(f"🌐 Deploy URL: {DEPLOYMENT_URL}")
        
        if st.button("🧪 Test Connection"):
            with st.spinner("Testing..."):
                test_ai = SmartAI()
                result = test_ai.call_llm("Say 'Hello' in 5 words")
                if "⚠️" in result:
                    st.error(result)
                else:
                    st.success(f"✅ {result}")
   
    # Create New Chatbot
    with st.sidebar.expander("➕ Create Chatbot", expanded=True):
        name = st.text_input("Company Name", key="new_company")
        url = st.text_input("Website URL", key="new_url")
       
        if st.button("🚀 Create Chatbot", type="primary"):
            if name and url:
                chatbot_id = hashlib.md5(f"{name}{url}{time.time()}".encode()).hexdigest()[:12]
                slug = re.sub(r'[^a-z0-9]+', '-', name.lower())
               
                progress = st.progress(0)
                status = st.empty()
               
                def cb(done, total, url_str):
                    progress.progress(done/total)
                    status.text(f"{done}/{total}: {url_str[:40]}...")
               
                bot = UniversalChatbot(name, url, chatbot_id)
                if bot.initialize(cb):
                    st.session_state.chatbots[slug] = bot
                    st.session_state.current_company = slug
                    st.session_state.chat_history = []
                    st.session_state.question_count = 0
                    st.session_state.lead_captured = False
                    st.session_state.lead_capture_mode = None
                    st.session_state.lead_data = {}
                   
                    embed = generate_embed_code(chatbot_id, name)
                    storage.save_chatbot(chatbot_id, name, url, embed)
                    st.success("✅ Chatbot created!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize")
   
    # List Chatbots
    if st.session_state.chatbots:
        st.sidebar.subheader("📋 Your Chatbots")
        for slug, bot in st.session_state.chatbots.items():
            col1, col2 = st.sidebar.columns([3,1])
            with col1:
                if st.button(f"💬 {bot.company_name}", key=f"sel_{slug}"):
                    st.session_state.current_company = slug
                    st.session_state.chat_history = []
                    st.session_state.question_count = 0
                    st.session_state.lead_captured = False
                    st.session_state.lead_capture_mode = None
                    st.session_state.lead_data = {}
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{slug}"):
                    del st.session_state.chatbots[slug]
                    if st.session_state.current_company == slug:
                        st.session_state.current_company = None
                    st.rerun()
   
    # View Leads Button
    if st.sidebar.button("📊 View All Leads"):
        st.subheader("📊 Captured Leads")
        st.info("💾 Leads stored in memory (will reset on restart)")
        
        leads = storage.get_leads()
        if leads:
            for lead in leads:
                with st.expander(f"🎯 {lead['username']} - {lead['company_name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Email:** {lead['mailid']}")
                        st.write(f"**Phone:** {lead['phonenumber']}")
                    with col2:
                        st.write(f"**Questions:** {lead['questions_asked']}")
                        st.write(f"**Time:** {lead['timestart'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Session:** {lead['session_id']}")
        else:
            st.info("No leads captured yet")
        return
   
    # Main Interface
    if not st.session_state.current_company:
        st.info("👈 Create a chatbot in the sidebar to get started!")
        
        # Instructions
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1️⃣ Create Chatbot
            - Enter company name
            - Provide website URL
            - AI scrapes & learns
            """)
        
        with col2:
            st.markdown("""
            #### 2️⃣ Get Embed Code
            - Copy HTML widget code
            - Multiple platform options
            - One-click deployment
            """)
        
        with col3:
            st.markdown("""
            #### 3️⃣ Capture Leads
            - Auto-trigger after 3 questions
            - Collect name, email, phone
            - View in dashboard
            """)
        
        return
   
    # Active Chatbot Interface
    bot = st.session_state.chatbots[st.session_state.current_company]
   
    # Header with metrics
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.subheader(f"💬 {bot.company_name}")
    with col2:
        st.metric("Questions", st.session_state.question_count)
    with col3:
        if st.session_state.lead_captured:
            st.success("✅ Lead Captured")
        else:
            st.info("🎯 Lead Pending")
   
    # Deployment Section
    with st.expander("🚀 Deploy to Your Website", expanded=False):
        st.markdown("### Choose Your Platform")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌐 Universal", 
            "📝 WordPress", 
            "🛒 Shopify", 
            "🎨 Wix",
            "⚙️ Custom"
        ])
        
        with tab1:
            st.markdown("#### Universal HTML Embed Code")
            st.caption("Works on any website - just paste before </body> tag")
            code = generate_embed_code(bot.chatbot_id, bot.company_name)
            st.code(code, language='html')
            st.download_button(
                "📥 Download HTML Widget",
                code,
                f"{bot.company_name}_widget.html",
                "text/html"
            )
        
        with tab2:
            st.markdown("#### WordPress Integration")
            st.caption("Add to theme files or use Custom HTML widget")
            wp_code = generate_wordpress_embed(bot.chatbot_id, bot.company_name)
            st.code(wp_code, language='html')
            
            with st.expander("📖 WordPress Instructions"):
                st.markdown("""
                **Method 1: Theme Editor**
                1. Go to Appearance → Theme Editor
                2. Open `footer.php`
                3. Paste code before `</body>`
                4. Click "Update File"
                
                **Method 2: Widget**
                1. Go to Appearance → Widgets
                2. Add "Custom HTML" widget
                3. Paste code
                4. Save
                
                **Method 3: Plugin**
                1. Install "Insert Headers and Footers" plugin
                2. Go to Settings → Insert Headers and Footers
                3. Paste in "Footer" section
                4. Save
                """)
        
        with tab3:
            st.markdown("#### Shopify Integration")
            st.caption("Add to your theme's liquid file")
            shopify_code = generate_shopify_embed(bot.chatbot_id, bot.company_name)
            st.code(shopify_code, language='html')
            
            with st.expander("📖 Shopify Instructions"):
                st.markdown("""
                1. Go to **Online Store** → **Themes**
                2. Click **Actions** → **Edit Code**
                3. Find and open `theme.liquid`
                4. Scroll to bottom, find `</body>` tag
                5. Paste code above `</body>`
                6. Click **Save**
                
                **Test**: Visit your store and you'll see the chat button!
                """)
        
        with tab4:
            st.markdown("#### Wix Integration")
            st.caption("Use Wix's embed widget feature")
            wix_code = generate_wix_embed(bot.chatbot_id, bot.company_name)
            st.text_area("Instructions & Code", wix_code, height=300)
            
            with st.expander("📖 Detailed Wix Guide"):
                st.markdown("""
                1. Open your Wix Editor
                2. Click the **+** (Add) button on the left
                3. Choose **Embed** → **Custom Embeds** → **Embed a Widget**
                4. In the popup, click **Add Custom Code**
                5. Paste the embed code
                6. Name it "AI Chatbot"
                7. Click **Update**
                8. Position and resize the widget area
                9. Publish your site
                """)
        
        with tab5:
            st.markdown("#### Custom Integration Options")
            
            st.markdown("**Direct iframe embed:**")
            iframe_code = f'<iframe src="{DEPLOYMENT_URL}?mode=embed&id={bot.chatbot_id}" width="400" height="600" frameborder="0"></iframe>'
            st.code(iframe_code, language='html')
            
            st.markdown("**JavaScript API access:**")
            st.code(f"""
// Initialize chatbot
const chatbot = {{
  id: '{bot.chatbot_id}',
  url: '{DEPLOYMENT_URL}',
  open: function() {{
    // Open chatbot programmatically
  }},
  close: function() {{
    // Close chatbot
  }}
}};
            """, language='javascript')
            
            st.markdown("**React Component:**")
            st.code(f"""
import React from 'react';

function Chatbot() {{
  return (
    <iframe
      src="{DEPLOYMENT_URL}?mode=embed&id={bot.chatbot_id}"
      style={{{{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '400px',
        height: '600px',
        border: 'none',
        borderRadius: '12px',
        boxShadow: '0 8px 30px rgba(0,0,0,0.3)'
      }}}}
    />
  );
}}

export default Chatbot;
            """, language='jsx')
   
    # Chat Interface
    st.markdown("---")
    st.markdown("### 💬 Test Chat Interface")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
   
    # Lead Capture Form
    if st.session_state.lead_capture_mode and not st.session_state.lead_captured:
        st.markdown("---")
        st.markdown("### 🎯 Let's Connect!")
        st.caption("Help us serve you better")
       
        if st.session_state.lead_capture_mode == 'ask_name':
            name = st.text_input("Your Name", key="name_input", placeholder="John Doe")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Continue", type="primary", key="submit_name"):
                    if name and name.strip():
                        st.session_state.lead_data['name'] = name.strip()
                        st.session_state.lead_capture_mode = 'ask_email'
                        st.rerun()
                    else:
                        st.error("Please enter your name")
            with col2:
                if st.button("⏭️ Skip", key="skip_name"):
                    st.session_state.lead_data['name'] = "Anonymous"
                    st.session_state.lead_capture_mode = 'ask_email'
                    st.rerun()
       
        elif st.session_state.lead_capture_mode == 'ask_email':
            email = st.text_input("Email Address", key="email_input", 
                                placeholder="john@example.com")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Continue", type="primary", key="submit_email"):
                    if email and validate_email(email):
                        st.session_state.lead_data['email'] = email.strip()
                        st.session_state.lead_capture_mode = 'ask_phone'
                        st.rerun()
                    else:
                        st.error("Please enter a valid email")
            with col2:
                if st.button("⏭️ Skip", key="skip_email"):
                    st.session_state.lead_data['email'] = "not_provided@example.com"
                    st.session_state.lead_capture_mode = 'ask_phone'
                    st.rerun()
       
        elif st.session_state.lead_capture_mode == 'ask_phone':
            phone = st.text_input("Phone Number (Optional)", key="phone_input",
                                placeholder="+1 234 567 8900")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Submit", type="primary", key="submit_phone"):
                    phone_value = phone.strip() if phone else "Not provided"
                    st.session_state.lead_data['phone'] = phone_value
                   
                    with st.spinner("Saving your information..."):
                        success = storage.save_lead(
                            bot.chatbot_id,
                            bot.company_name,
                            st.session_state.lead_data.get('name', 'Anonymous'),
                            st.session_state.lead_data.get('email', 'not_provided@example.com'),
                            phone_value,
                            st.session_state.session_id,
                            st.session_state.question_count,
                            st.session_state.chat_history
                        )
                   
                    if success:
                        st.session_state.lead_captured = True
                        st.session_state.lead_capture_mode = None
                        st.balloons()
                        st.success("✅ Thank you! Continuing our conversation...")
                        time.sleep(1)
                        st.rerun()
           
            with col2:
                if st.button("⏭️ Skip", key="skip_phone"):
                    with st.spinner("Saving..."):
                        success = storage.save_lead(
                            bot.chatbot_id,
                            bot.company_name,
                            st.session_state.lead_data.get('name', 'Anonymous'),
                            st.session_state.lead_data.get('email', 'not_provided@example.com'),
                            "Not provided",
                            st.session_state.session_id,
                            st.session_state.question_count,
                            st.session_state.chat_history
                        )
                   
                    if success:
                        st.session_state.lead_captured = True
                        st.session_state.lead_capture_mode = None
                        st.balloons()
                        st.success("✅ Thank you!")
                        time.sleep(1)
                        st.rerun()
   
    # Chat Input
    if question := st.chat_input("Ask anything about the company...",
                                  disabled=bool(st.session_state.lead_capture_mode and 
                                              not st.session_state.lead_captured)):
        if st.session_state.lead_capture_mode and not st.session_state.lead_captured:
            st.warning("⚠️ Please complete the form above to continue chatting")
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": question})
           
            # Generate AI response
            with st.spinner("Thinking..."):
                answer = bot.ask(question)
           
            # Add assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.question_count += 1
           
            # Trigger lead capture after 3 questions
            if (st.session_state.question_count >= 3 and 
                not st.session_state.lead_captured and 
                not st.session_state.lead_capture_mode):
                st.session_state.lead_capture_mode = 'ask_name'
           
            st.rerun()


if __name__ == "__main__":
    main()
