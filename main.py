import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import hashlib
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """Central configuration management"""
    
    # API Configuration
    OPENROUTER_API_KEY = ""  # Add your key here or use .env
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
    
    # Model Selection
    MODELS = {
        "free": "kwaipilot/kat-coder-pro:free",
        "cheap": "anthropic/claude-3-haiku",
        "smart": "anthropic/claude-3-sonnet"
    }
    CURRENT_MODEL = "free"  # Change to "cheap" or "smart" for better results
    
    # Deployment
    DEPLOYMENT_URL = "http://localhost:8501"
    
    # Scraping Settings
    SCRAPE_TIMEOUT = 8
    MAX_PAGES = 15
    MAX_CONTENT_LENGTH = 5000
    
    # Lead Capture Settings
    QUESTIONS_BEFORE_CAPTURE = 3
    
    # UI Settings
    APP_TITLE = "🤖 AI Chatbot Lead Generator Pro"
    APP_ICON = "🤖"
    
    @classmethod
    def get_model(cls):
        return cls.MODELS.get(cls.CURRENT_MODEL, cls.MODELS["free"])
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        issues = []
        
        if not cls.OPENROUTER_API_KEY:
            issues.append("❌ API Key not set")
        elif not cls.OPENROUTER_API_KEY.startswith("sk-or-v1-"):
            issues.append("⚠️ API Key format looks incorrect")
        
        return issues


# ==========================================
# PERSISTENT STORAGE (ARTIFACT-COMPATIBLE)
# ==========================================

class PersistentStorage:
    """Storage using Streamlit session state with persistence"""
    
    def __init__(self):
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage in session state"""
        if 'storage_leads' not in st.session_state:
            st.session_state.storage_leads = []
        if 'storage_chatbots' not in st.session_state:
            st.session_state.storage_chatbots = {}
        if 'storage_next_lead_id' not in st.session_state:
            st.session_state.storage_next_lead_id = 1
    
    def save_lead(self, chatbot_id: str, company_name: str, user_name: str,
                  user_email: str, user_phone: str, session_id: str,
                  questions_asked: int, conversation: List[Dict]) -> bool:
        """Save a lead"""
        try:
            lead = {
                'id': st.session_state.storage_next_lead_id,
                'chatbot_id': chatbot_id,
                'company_name': company_name,
                'user_name': user_name or "Anonymous",
                'user_email': user_email or "not_provided@example.com",
                'user_phone': user_phone or "Not provided",
                'session_id': session_id,
                'questions_asked': questions_asked,
                'conversation': json.dumps(conversation),
                'created_at': datetime.now().isoformat(),
                'status': 'new'
            }
            
            st.session_state.storage_leads.append(lead)
            st.session_state.storage_next_lead_id += 1
            
            print(f"[Storage] ✅ Lead saved: {user_name} for {company_name}")
            return True
            
        except Exception as e:
            print(f"[Storage] ❌ Save lead error: {e}")
            return False
    
    def get_leads(self, chatbot_id: Optional[str] = None) -> List[Dict]:
        """Get all leads or leads for specific chatbot"""
        try:
            if chatbot_id:
                return [l for l in st.session_state.storage_leads 
                       if l['chatbot_id'] == chatbot_id]
            return st.session_state.storage_leads
        except Exception as e:
            print(f"[Storage] ❌ Get leads error: {e}")
            return []
    
    def save_chatbot(self, chatbot_id: str, company_name: str,
                     website_url: str, embed_code: str) -> bool:
        """Save chatbot configuration"""
        try:
            st.session_state.storage_chatbots[chatbot_id] = {
                'chatbot_id': chatbot_id,
                'company_name': company_name,
                'website_url': website_url,
                'embed_code': embed_code,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'total_leads': 0,
                'total_questions': 0
            }
            
            print(f"[Storage] ✅ Chatbot saved: {company_name}")
            return True
            
        except Exception as e:
            print(f"[Storage] ❌ Save chatbot error: {e}")
            return False
    
    def get_chatbot(self, chatbot_id: str) -> Optional[Dict]:
        """Get chatbot by ID"""
        return st.session_state.storage_chatbots.get(chatbot_id)
    
    def update_chatbot_stats(self, chatbot_id: str, questions_inc: int = 0,
                            leads_inc: int = 0):
        """Update chatbot statistics"""
        if chatbot_id in st.session_state.storage_chatbots:
            bot = st.session_state.storage_chatbots[chatbot_id]
            bot['total_questions'] = bot.get('total_questions', 0) + questions_inc
            bot['total_leads'] = bot.get('total_leads', 0) + leads_inc
    
    def delete_chatbot(self, chatbot_id: str) -> bool:
        """Delete a chatbot"""
        try:
            if chatbot_id in st.session_state.storage_chatbots:
                del st.session_state.storage_chatbots[chatbot_id]
                return True
            return False
        except Exception as e:
            print(f"[Storage] ❌ Delete error: {e}")
            return False
    
    def export_leads_csv(self, chatbot_id: Optional[str] = None) -> str:
        """Export leads as CSV"""
        leads = self.get_leads(chatbot_id)
        
        if not leads:
            return ""
        
        csv_lines = ["ID,Company,Name,Email,Phone,Questions,Date,Status"]
        
        for lead in leads:
            csv_lines.append(
                f"{lead['id']},"
                f"{lead['company_name']},"
                f"{lead['user_name']},"
                f"{lead['user_email']},"
                f"{lead['user_phone']},"
                f"{lead['questions_asked']},"
                f"{lead['created_at']},"
                f"{lead['status']}"
            )
        
        return "\n".join(csv_lines)


# ==========================================
# SMART WEB SCRAPER
# ==========================================

class SmartScraper:
    """Enhanced web scraper with better error handling"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.timeout = Config.SCRAPE_TIMEOUT
        self.max_pages = Config.MAX_PAGES
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape a single page"""
        try:
            print(f"[Scraper] 🌐 Fetching: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"[Scraper] ⚠️ Status {response.status_code} for {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            # Extract text
            content = soup.get_text(separator='\n', strip=True)
            
            # Clean and filter
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if len(line) > 20 and len(line) < 500:  # Filter noise
                    lines.append(line)
            
            if not lines:
                return None
            
            # Limit content
            text = '\n'.join(lines[:100])[:Config.MAX_CONTENT_LENGTH]
            
            return {
                "url": url,
                "content": text,
                "length": len(text)
            }
            
        except requests.exceptions.Timeout:
            print(f"[Scraper] ⏱️ Timeout: {url}")
            return None
        except Exception as e:
            print(f"[Scraper] ❌ Error: {e}")
            return None
    
    def scrape_website(self, base_url: str, 
                      progress_callback=None) -> Tuple[List[Dict], Dict]:
        """Scrape multiple pages from a website"""
        
        # Normalize URL
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
        
        # Generate URLs to scrape
        urls = [
            base_url,
            f"{base_url}/about",
            f"{base_url}/services",
            f"{base_url}/products",
            f"{base_url}/contact"
        ][:self.max_pages]
        
        pages = []
        all_text = ""
        
        # Concurrent scraping
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self.scrape_page, url): url 
                      for url in urls}
            
            for i, future in enumerate(as_completed(futures), 1):
                if progress_callback:
                    progress_callback(i, len(urls), futures[future])
                
                result = future.result()
                if result:
                    pages.append(result)
                    all_text += "\n" + result['content']
        
        # Extract contact info
        emails = list(set(re.findall(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            all_text
        )))[:3]
        
        phones = list(set(re.findall(
            r'\+?[\d\s\-\(\)]{10,}',
            all_text
        )))[:3]
        
        contact_info = {
            "emails": emails,
            "phones": phones
        }
        
        print(f"[Scraper] ✅ Scraped {len(pages)} pages, "
              f"found {len(emails)} emails, {len(phones)} phones")
        
        return pages, contact_info


# ==========================================
# AI ENGINE
# ==========================================

class AIEngine:
    """Enhanced AI engine with caching and error handling"""
    
    def __init__(self):
        self.cache = {}
        self.api_calls = 0
        self.api_errors = 0
        self.last_error = None
    
    def _cache_key(self, prompt: str) -> str:
        """Generate cache key"""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]
    
    def call_api(self, prompt: str, max_retries: int = 2) -> str:
        """Call OpenRouter API with retry logic"""
        
        # Validate API key
        if not Config.OPENROUTER_API_KEY:
            return self._format_error("no_api_key")
        
        # Check cache
        cache_key = self._cache_key(prompt)
        if cache_key in self.cache:
            print(f"[AI] 💾 Cache hit")
            return self.cache[cache_key]
        
        # Make API call
        for attempt in range(max_retries):
            try:
                self.api_calls += 1
                print(f"[AI] 🚀 API call #{self.api_calls} (attempt {attempt + 1})")
                
                response = requests.post(
                    Config.OPENROUTER_API_BASE,
                    headers={
                        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": Config.DEPLOYMENT_URL,
                        "X-Title": "AI Chatbot Lead Generator"
                    },
                    json={
                        "model": Config.get_model(),
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 600,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                
                # Handle response
                if response.status_code == 200:
                    data = response.json()
                    
                    if "choices" in data and data["choices"]:
                        answer = data["choices"][0]["message"]["content"].strip()
                        self.cache[cache_key] = answer
                        print(f"[AI] ✅ Success ({len(answer)} chars)")
                        return answer
                    
                    return "I received an unexpected response. Please try again."
                
                elif response.status_code == 401:
                    self.api_errors += 1
                    return self._format_error("auth_failed")
                
                elif response.status_code == 402:
                    self.api_errors += 1
                    return self._format_error("no_credits")
                
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"[AI] ⏱️ Rate limited, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    self.api_errors += 1
                    return self._format_error("rate_limit")
                
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    self.api_errors += 1
                    self.last_error = response.text[:200]
                    return self._format_error("api_error", response.status_code)
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "⏱️ Request timed out. Please try again."
            
            except Exception as e:
                print(f"[AI] ❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"❌ Error: {type(e).__name__}"
        
        return "Failed after multiple attempts. Please try again."
    
    def _format_error(self, error_type: str, status_code: int = None) -> str:
        """Format error messages"""
        
        errors = {
            "no_api_key": """❌ **API Key Not Set**

Please add your OpenRouter API key:
1. Get key from: https://openrouter.ai/keys
2. Add to Config.OPENROUTER_API_KEY in code
3. Or create .env file with OPENROUTER_API_KEY=your_key
4. Restart the app""",
            
            "auth_failed": """❌ **Authentication Failed**

Your API key is invalid or expired.
1. Check key at: https://openrouter.ai/keys
2. Generate new key if needed
3. Update Config.OPENROUTER_API_KEY
4. Restart app""",
            
            "no_credits": """💰 **Insufficient Credits**

Add credits to your OpenRouter account:
1. Go to: https://openrouter.ai/credits
2. Add at least $5
3. Wait 1-2 minutes
4. Try again""",
            
            "rate_limit": """⏱️ **Rate Limit Exceeded**

Too many requests. Try:
1. Wait 30 seconds
2. Use paid model (change CURRENT_MODEL)
3. Add more credits""",
            
            "api_error": f"""⚠️ **API Error {status_code}**

Something went wrong. Try:
1. Wait a few seconds
2. Check OpenRouter status
3. Try different model"""
        }
        
        return errors.get(error_type, "Unknown error occurred")
    
    def get_stats(self) -> Dict:
        """Get AI statistics"""
        return {
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "cache_size": len(self.cache),
            "last_error": self.last_error
        }


# ==========================================
# CHATBOT CORE
# ==========================================

class SmartChatbot:
    """Enhanced chatbot with context awareness"""
    
    def __init__(self, chatbot_id: str, company_name: str, website_url: str):
        self.chatbot_id = chatbot_id
        self.company_name = company_name
        self.website_url = website_url
        self.pages = []
        self.contact_info = {}
        self.ready = False
        self.ai = AIEngine()
        self.scraper = SmartScraper()
    
    def initialize(self, progress_callback=None) -> bool:
        """Initialize chatbot by scraping website"""
        try:
            print(f"[Chatbot] 🔧 Initializing: {self.company_name}")
            
            self.pages, self.contact_info = self.scraper.scrape_website(
                self.website_url, 
                progress_callback
            )
            
            if not self.pages:
                print(f"[Chatbot] ⚠️ No content scraped")
                return False
            
            self.ready = True
            print(f"[Chatbot] ✅ Ready: {self.company_name}")
            return True
            
        except Exception as e:
            print(f"[Chatbot] ❌ Init error: {e}")
            return False
    
    def ask(self, question: str) -> str:
        """Process a question"""
        
        if not self.ready:
            return "⚠️ Chatbot is initializing. Please wait..."
        
        # Handle greetings
        if any(g in question.lower() for g in ['hi', 'hello', 'hey', 'greetings']):
            return f"👋 Hello! I'm the AI assistant for **{self.company_name}**. How can I help you today?"
        
        # Handle contact info requests
        if any(k in question.lower() for k in ['email', 'contact', 'phone', 'reach']):
            return self._format_contact_info()
        
        # Handle general questions with AI
        return self._generate_ai_response(question)
    
    def _format_contact_info(self) -> str:
        """Format contact information"""
        msg = f"📞 **Contact {self.company_name}**\n\n"
        
        if self.contact_info.get('emails'):
            msg += "📧 **Email:** " + ", ".join(self.contact_info['emails']) + "\n\n"
        
        if self.contact_info.get('phones'):
            msg += "📱 **Phone:** " + ", ".join(self.contact_info['phones']) + "\n\n"
        
        msg += f"🌐 **Website:** {self.website_url}"
        
        return msg
    
    def _generate_ai_response(self, question: str) -> str:
        """Generate AI response using scraped context"""
        
        # Prepare context from scraped pages
        context_parts = []
        for page in self.pages[:3]:  # Use top 3 pages
            context_parts.append(page['content'][:1000])
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant for {self.company_name}.

Website Context:
{context}

User Question: {question}

Provide a helpful, natural 2-3 sentence answer based on the context. If the question cannot be answered from the context, provide a general helpful response and suggest contacting the company directly.

Answer:"""
        
        return self.ai.call_api(prompt)


# ==========================================
# EMBED CODE GENERATOR
# ==========================================

def generate_embed_code(chatbot_id: str, company_name: str) -> str:
    """Generate embeddable widget code"""
    
    embed_url = f"{Config.DEPLOYMENT_URL}?mode=embed&id={chatbot_id}"
    
    return f'''<!-- {company_name} AI Chatbot Widget -->
<div id="ai-chatbot-{chatbot_id}"></div>
<script>
(function() {{
  var config = {{
    embedUrl: '{embed_url}',
    position: 'bottom-right',
    primaryColor: '#667eea',
    chatIcon: '💬'
  }};
  
  // Create chat button
  var btn = document.createElement('button');
  btn.innerHTML = config.chatIcon + ' Chat';
  btn.style.cssText = 'position:fixed;bottom:20px;right:20px;background:linear-gradient(135deg,' + config.primaryColor + ',#764ba2);color:white;border:none;border-radius:50px;padding:15px 30px;font-size:16px;font-weight:600;cursor:pointer;box-shadow:0 4px 15px rgba(102,126,234,0.4);z-index:999999;transition:all 0.3s ease;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;';
  
  btn.onmouseenter = function() {{
    this.style.transform = 'scale(1.05)';
    this.style.boxShadow = '0 6px 20px rgba(102,126,234,0.5)';
  }};
  
  btn.onmouseleave = function() {{
    this.style.transform = 'scale(1)';
    this.style.boxShadow = '0 4px 15px rgba(102,126,234,0.4)';
  }};
  
  // Create chat container
  var container = document.createElement('div');
  container.style.cssText = 'position:fixed;bottom:90px;right:20px;width:400px;height:600px;max-width:calc(100vw - 40px);max-height:calc(100vh - 120px);border-radius:16px;box-shadow:0 8px 30px rgba(0,0,0,0.3);z-index:999998;display:none;background:white;overflow:hidden;';
  
  // Create iframe
  var iframe = document.createElement('iframe');
  iframe.src = config.embedUrl;
  iframe.style.cssText = 'width:100%;height:100%;border:none;border-radius:16px;';
  iframe.setAttribute('allow', 'clipboard-write');
  
  container.appendChild(iframe);
  
  // Toggle function
  btn.onclick = function() {{
    var isOpen = container.style.display !== 'none';
    container.style.display = isOpen ? 'none' : 'block';
    btn.innerHTML = isOpen ? (config.chatIcon + ' Chat') : '✕ Close';
  }};
  
  // Append to body
  document.body.appendChild(btn);
  document.body.appendChild(container);
  
  console.log('AI Chatbot Widget loaded for {company_name}');
}})();
</script>

<!-- Optional: Add custom styling -->
<style>
  #ai-chatbot-{chatbot_id} {{
    /* Add custom styles here */
  }}
</style>'''


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def validate_email(email: str) -> bool:
    """Validate email format"""
    if not email or not email.strip():
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Validate URL format"""
    if not url:
        return False
    pattern = r'^(https?://)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(/.*)?$'
    return bool(re.match(pattern, url))

def generate_chatbot_id(company_name: str, website_url: str) -> str:
    """Generate unique chatbot ID"""
    unique_string = f"{company_name}{website_url}{time.time()}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'storage': PersistentStorage(),
        'active_chatbots': {},
        'current_chatbot_id': None,
        'chat_history': [],
        'question_count': 0,
        'lead_capture_step': None,
        'lead_data': {},
        'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16],
        'lead_captured': False,
        'view_mode': 'chat',  # 'chat', 'leads', 'settings'
        
        # Embed mode states
        'embed_mode': False,
        'embed_chat_history': [],
        'embed_question_count': 0,
        'embed_lead_captured': False,
        'embed_lead_step': None,
        'embed_lead_data': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_embed_mode():
    """Render chatbot in embed mode"""
    
    st.set_page_config(
        page_title="AI Chat",
        page_icon="💬",
        layout="wide"
    )
    
    # Minimal styling for embed
    st.markdown("""
    <style>
        .stApp {
            background: transparent;
        }
        .main .block-container {
            padding: 1rem;
            max-width: 100%;
        }
        header, footer, .stDeployButton {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Get chatbot ID
    chatbot_id = st.query_params.get("id")
    
    if not chatbot_id:
        st.error("⚠️ Invalid chatbot configuration")
        return
    
    # Load chatbot
    storage = st.session_state.storage
    bot_config = storage.get_chatbot(chatbot_id)
    
    if not bot_config:
        st.error("⚠️ Chatbot not found")
        return
    
    # Initialize chatbot if needed
    if chatbot_id not in st.session_state.active_chatbots:
        bot = SmartChatbot(
            chatbot_id,
            bot_config['company_name'],
            bot_config['website_url']
        )
        
        with st.spinner("🔄 Loading..."):
            if bot.initialize():
                st.session_state.active_chatbots[chatbot_id] = bot
            else:
                st.error("Failed to initialize chatbot")
                return
    
    bot = st.session_state.active_chatbots[chatbot_id]
    
    # Header
    st.markdown(f"### 💬 {bot.company_name}")
    st.caption("AI-powered assistant")
    
    # Display chat history
    for msg in st.session_state.embed_chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Lead capture (simplified for embed)
    if st.session_state.embed_lead_step and not st.session_state.embed_lead_captured:
        st.info("📋 Please share your email to continue")
        
        email = st.text_input("Email Address", key="embed_email_input")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Submit", type="primary"):
                if validate_email(email):
                    storage.save_lead(
                        chatbot_id,
                        bot.company_name,
                        "Website Visitor",
                        email,
                        "Not provided",
                        st.session_state.session_id,
                        st.session_state.embed_question_count,
                        st.session_state.embed_chat_history
                    )
                    st.session_state.embed_lead_captured = True
                    st.session_state.embed_lead_step = None
                    st.success("✅ Thank you!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please enter a valid email")
        
        with col2:
            if st.button("Skip"):
                st.session_state.embed_lead_step = None
                st.rerun()
    
    # Chat input
    if question := st.chat_input("Type your message..."):

# CONTINUATION OF CHATBOT APPLICATION
# This continues from the previous artifact

# Complete the render_embed_mode function
def render_embed_mode_complete():
    """Complete embed mode rendering (continuation)"""
    
    # ... (previous code from embed mode)
    
    # Chat input
    if question := st.chat_input("Type your message...", 
                                  disabled=bool(st.session_state.embed_lead_step)):
        # Add to history
        st.session_state.embed_chat_history.append({
            "role": "user",
            "content": question
        })
        
        # Get response
        with st.spinner("🤔 Thinking..."):
            answer = bot.ask(question)
        
        # Add response to history
        st.session_state.embed_chat_history.append({
            "role": "assistant",
            "content": answer
        })
        
        st.session_state.embed_question_count += 1
        
        # Trigger lead capture after N questions
        if (st.session_state.embed_question_count >= Config.QUESTIONS_BEFORE_CAPTURE 
            and not st.session_state.embed_lead_captured):
            st.session_state.embed_lead_step = 'ask_email'
        
        # Update stats
        storage.update_chatbot_stats(chatbot_id, questions_inc=1)
        
        st.rerun()

def render_admin_panel():
    """Render admin control panel"""
    
    st.sidebar.title("🎛️ Control Panel")
    
    storage = st.session_state.storage
    
    # Configuration Status
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        config_issues = Config.validate()
        
        if config_issues:
            for issue in config_issues:
                st.warning(issue)
            
            st.markdown("""
            **Quick Fix:**
            1. Get API key from [OpenRouter](https://openrouter.ai/keys)
            2. Update `Config.OPENROUTER_API_KEY` in code
            3. Restart the app
            """)
        else:
            st.success("✅ Configuration valid")
        
        st.caption(f"Model: {Config.get_model()}")
        st.caption(f"Deployment: {Config.DEPLOYMENT_URL}")
    
    # Create New Chatbot
    with st.sidebar.expander("➕ Create Chatbot", expanded=True):
        with st.form("create_chatbot_form"):
            company_name = st.text_input(
                "Company Name",
                placeholder="Acme Corporation"
            )
            
            website_url = st.text_input(
                "Website URL",
                placeholder="acme.com or https://acme.com"
            )
            
            submitted = st.form_submit_button("🚀 Create Chatbot", type="primary")
            
            if submitted:
                if not company_name or not website_url:
                    st.error("Please fill all fields")
                elif not validate_url(website_url):
                    st.error("Please enter a valid URL")
                else:
                    # Generate chatbot ID
                    chatbot_id = generate_chatbot_id(company_name, website_url)
                    
                    # Create chatbot
                    bot = SmartChatbot(chatbot_id, company_name, website_url)
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, url):
                        progress_bar.progress(current / total)
                        status_text.text(f"Scraping {current}/{total}: {url}")
                    
                    # Initialize
                    if bot.initialize(progress_callback):
                        # Save to storage
                        st.session_state.active_chatbots[chatbot_id] = bot
                        
                        # Generate embed code
                        embed_code = generate_embed_code(chatbot_id, company_name)
                        
                        # Save to database
                        storage.save_chatbot(chatbot_id, company_name, 
                                            website_url, embed_code)
                        
                        # Set as current
                        st.session_state.current_chatbot_id = chatbot_id
                        st.session_state.chat_history = []
                        st.session_state.question_count = 0
                        st.session_state.lead_captured = False
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Chatbot created successfully!")
                        
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to create chatbot. Please check the website URL.")
    
    # List Existing Chatbots
    chatbots = storage.storage_chatbots
    
    if chatbots:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Your Chatbots")
        
        for bot_id, bot_config in chatbots.items():
            col1, col2 = st.sidebar.columns([4, 1])
            
            with col1:
                if st.button(
                    f"💬 {bot_config['company_name']}",
                    key=f"select_{bot_id}",
                    use_container_width=True
                ):
                    st.session_state.current_chatbot_id = bot_id
                    st.session_state.chat_history = []
                    st.session_state.question_count = 0
                    st.session_state.view_mode = 'chat'
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{bot_id}"):
                    storage.delete_chatbot(bot_id)
                    if bot_id in st.session_state.active_chatbots:
                        del st.session_state.active_chatbots[bot_id]
                    if st.session_state.current_chatbot_id == bot_id:
                        st.session_state.current_chatbot_id = None
                    st.rerun()
    
    # View Mode Selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 View")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.view_mode = 'chat'
            st.rerun()
    
    with col2:
        if st.button("📊 Leads", use_container_width=True):
            st.session_state.view_mode = 'leads'
            st.rerun()


def render_chat_interface():
    """Render main chat interface"""
    
    storage = st.session_state.storage
    chatbot_id = st.session_state.current_chatbot_id
    
    if not chatbot_id:
        # Welcome screen
        st.markdown("## 👋 Welcome to AI Chatbot Lead Generator")
        
        st.markdown("""
        ### 🚀 Get Started in 3 Steps:
        
        1. **Create a Chatbot** - Enter your company name and website
        2. **Test It** - Chat with your AI assistant
        3. **Deploy** - Get embed code for your website
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### ✨ Features
            - AI-powered responses
            - Automatic website learning
            - Lead capture forms
            - Embed on any website
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Benefits
            - 24/7 customer support
            - Capture qualified leads
            - Reduce support costs
            - Improve engagement
            """)
        
        with col3:
            st.markdown("""
            #### 🎯 Use Cases
            - E-commerce support
            - Service inquiries
            - Lead qualification
            - FAQ automation
            """)
        
        return
    
    # Load chatbot
    bot_config = storage.get_chatbot(chatbot_id)
    
    if not bot_config:
        st.error("Chatbot not found")
        return
    
    # Initialize if needed
    if chatbot_id not in st.session_state.active_chatbots:
        bot = SmartChatbot(
            chatbot_id,
            bot_config['company_name'],
            bot_config['website_url']
        )
        
        with st.spinner("Loading chatbot..."):
            if bot.initialize():
                st.session_state.active_chatbots[chatbot_id] = bot
            else:
                st.error("Failed to load chatbot")
                return
    
    bot = st.session_state.active_chatbots[chatbot_id]
    
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"### 💬 {bot.company_name}")
    
    with col2:
        st.metric("Questions", st.session_state.question_count)
    
    with col3:
        if st.session_state.lead_captured:
            st.success("✅ Lead Captured")
        else:
            st.info("🎯 No Lead Yet")
    
    # Deployment Section
    with st.expander("🚀 Deploy to Website", expanded=False):
        st.markdown("### Embed Code")
        st.markdown("Copy and paste this code into your website:")
        
        embed_code = bot_config['embed_code']
        st.code(embed_code, language='html')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.download_button(
                "📥 Download HTML",
                embed_code,
                file_name=f"{bot.company_name}_chatbot.html",
                mime="text/html"
            )
        
        with col2:
            if st.button("📋 Copy to Clipboard"):
                st.code(embed_code)
                st.info("Select and copy the code above")
        
        st.markdown("""
        #### 📍 Where to Add:
        
        **HTML Websites:**
        - Paste before `</body>` tag
        
        **WordPress:**
        - Use Custom HTML widget
        - Or add to theme footer
        
        **Shopify:**
        - Theme > Edit Code > theme.liquid
        - Paste before `</body>`
        
        **Other Platforms:**
        - Look for "Custom HTML" or "Embed Code" options
        """)
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Lead Capture Form
    if st.session_state.lead_capture_step and not st.session_state.lead_captured:
        st.markdown("---")
        st.markdown("### 🎯 Let's Connect!")
        
        if st.session_state.lead_capture_step == 'ask_name':
            name = st.text_input("Your Name", placeholder="John Doe")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("✅ Continue", type="primary"):
                    if name and name.strip():
                        st.session_state.lead_data['name'] = name.strip()
                        st.session_state.lead_capture_step = 'ask_email'
                        st.rerun()
                    else:
                        st.error("Please enter your name")
            
            with col2:
                if st.button("⏭️ Skip"):
                    st.session_state.lead_data['name'] = "Anonymous"
                    st.session_state.lead_capture_step = 'ask_email'
                    st.rerun()
        
        elif st.session_state.lead_capture_step == 'ask_email':
            email = st.text_input("Email Address", placeholder="john@example.com")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("✅ Continue", type="primary"):
                    if validate_email(email):
                        st.session_state.lead_data['email'] = email.strip()
                        st.session_state.lead_capture_step = 'ask_phone'
                        st.rerun()
                    else:
                        st.error("Please enter a valid email")
            
            with col2:
                if st.button("⏭️ Skip"):
                    st.session_state.lead_data['email'] = "not_provided@example.com"
                    st.session_state.lead_capture_step = 'ask_phone'
                    st.rerun()
        
        elif st.session_state.lead_capture_step == 'ask_phone':
            phone = st.text_input("Phone Number (Optional)", 
                                 placeholder="+1 234 567 8900")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("✅ Submit", type="primary"):
                    phone_value = phone.strip() if phone else "Not provided"
                    st.session_state.lead_data['phone'] = phone_value
                    
                    # Save lead
                    success = storage.save_lead(
                        chatbot_id,
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
                        st.session_state.lead_capture_step = None
                        storage.update_chatbot_stats(chatbot_id, leads_inc=1)
                        st.balloons()
                        st.success("✅ Thank you! We'll be in touch soon.")
                        time.sleep(2)
                        st.rerun()
            
            with col2:
                if st.button("⏭️ Skip"):
                    # Save without phone
                    success = storage.save_lead(
                        chatbot_id,
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
                        st.session_state.lead_capture_step = None
                        storage.update_chatbot_stats(chatbot_id, leads_inc=1)
                        st.success("✅ Thank you!")
                        time.sleep(2)
                        st.rerun()
    
    # Chat Input
    if question := st.chat_input(
        "Ask anything...",
        disabled=bool(st.session_state.lead_capture_step)
    ):
        # Add question to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        # Get answer
        with st.spinner("🤔 Thinking..."):
            answer = bot.ask(question)
        
        # Add answer to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
        
        st.session_state.question_count += 1
        
        # Trigger lead capture
        if (st.session_state.question_count >= Config.QUESTIONS_BEFORE_CAPTURE
            and not st.session_state.lead_captured
            and not st.session_state.lead_capture_step):
            st.session_state.lead_capture_step = 'ask_name'
        
        # Update stats
        storage.update_chatbot_stats(chatbot_id, questions_inc=1)
        
        st.rerun()


def render_leads_dashboard():
    """Render leads management dashboard"""
    
    storage = st.session_state.storage
    
    st.markdown("## 📊 Leads Dashboard")
    
    # Filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        chatbot_options = {"All Chatbots": None}
        for bot_id, bot_config in storage.storage_chatbots.items():
            chatbot_options[bot_config['company_name']] = bot_id
        
        selected_bot = st.selectbox(
            "Filter by Chatbot",
            options=list(chatbot_options.keys())
        )
        
        filter_chatbot_id = chatbot_options[selected_bot]
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col3:
        leads = storage.get_leads(filter_chatbot_id)
        csv_data = storage.export_leads_csv(filter_chatbot_id)
        
        if csv_data:
            st.download_button(
                "📥 Export CSV",
                csv_data,
                file_name=f"leads_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Leads summary
    total_leads = len(leads)
    
    if total_leads == 0:
        st.info("No leads captured yet. Start chatting to capture leads!")
        return
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Leads", total_leads)
    
    with col2:
        valid_emails = sum(1 for l in leads if '@' in l['user_email'] 
                          and 'not_provided' not in l['user_email'])
        st.metric("Valid Emails", valid_emails)
    
    with col3:
        with_phone = sum(1 for l in leads if l['user_phone'] != "Not provided")
        st.metric("With Phone", with_phone)
    
    with col4:
        avg_questions = sum(l['questions_asked'] for l in leads) / len(leads)
        st.metric("Avg Questions", f"{avg_questions:.1f}")
    
    st.markdown("---")
    
    # Leads list
    for lead in reversed(leads): # Show newest first
        with st.expander(
            f"🎯 {lead['user_name']} - {lead['company_name']} "
            f"({lead['created_at'][:10]})"
        ):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                **Contact Information:**
                - 👤 Name: {lead['user_name']}
                - 📧 Email: {lead['user_email']}
                - 📱 Phone: {lead['user_phone']}
                """)
            
            with col2:
                st.markdown(f"""
                **Engagement:**
                - 💬 Questions: {lead['questions_asked']}
                - 🕒 Date: {lead['created_at'][:16]}
                - 🆔 Session: {lead['session_id'][:8]}
                """)
            
            # Show conversation
            if lead.get('conversation'):
                with st.expander("📝 View Conversation"):
                    try:
                        conversation = json.loads(lead['conversation'])
                        for msg in conversation:
                            role_icon = "👤" if msg['role'] == 'user' else "🤖"
                            st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['content']}")
                    except:
                        st.text(lead['conversation'])


def main():
    """Main application entry point"""
    
    # Check for embed mode
    mode = st.query_params.get("mode")
    if mode == "embed":
        render_embed_mode()
        return
    
    # Configure page
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }""")
