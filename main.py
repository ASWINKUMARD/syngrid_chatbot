import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import os
import hashlib
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[ENV] Loaded environment variables from .env")
except ImportError:
    print("[ENV] python-dotenv not installed, using system environment")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "free": "kwaipilot/kat-coder-pro:free"
}

# Choose your model (change this if free model doesn't work)
CURRENT_MODEL = "free"  # Change to "cheap" for more reliability
MODEL = MODELS[CURRENT_MODEL]

DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL", "http://localhost:8501")

print("\n" + "="*70)
print("🤖 AI CHATBOT STARTUP DIAGNOSTICS")
print("="*70)

# Check API Key
if OPENROUTER_API_KEY:
    key_preview = f"{OPENROUTER_API_KEY[:15]}...{OPENROUTER_API_KEY[-8:]}"
    print(f"✅ API Key: {key_preview}")
    print(f"   Length: {len(OPENROUTER_API_KEY)} characters")
    
    if not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
        print("   ⚠️  WARNING: Key doesn't start with 'sk-or-v1-'")
        print("   This might be invalid. Get a new key from https://openrouter.ai/keys")
else:
    print("❌ API Key: NOT SET!")
    print("   Create a .env file with: OPENROUTER_API_KEY=your_key_here")

print(f"🤖 Model: {MODEL} ({'Free' if CURRENT_MODEL == 'free' else 'Paid'})")
print(f"🌐 Deployment URL: {DEPLOYMENT_URL}")
print("="*70 + "\n")

class InMemoryStorage:
    """Handles all data storage in memory"""
   
    def __init__(self):
        self.leads = []
        self.chatbots = {}
        self.next_lead_id = 1
        self.next_chatbot_id = 1
   
    def save_lead(self, chatbot_id, company_name, user_name, user_email,
                  user_phone, session_id, questions_asked, conversation):
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
            print(f"[Storage] ✅ Lead saved: {user_name}")
            return True
        except Exception as e:
            print(f"[Storage] ❌ Error: {e}")
            return False
   
    def get_leads(self, chatbot_id=None):
        try:
            if chatbot_id:
                return [l for l in self.leads if l['chatbot_id'] == chatbot_id]
            return self.leads
        except Exception as e:
            print(f"[Storage] ❌ Get leads error: {e}")
            return []
   
    def save_chatbot(self, chatbot_id, company_name, website_url, embed_code):
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
            print(f"[Storage] ❌ Error: {e}")
            return False
   
    def get_chatbot(self, chatbot_id):
        return self.chatbots.get(chatbot_id)
    
    def get_chatbot_by_id(self, chatbot_id):
        for bot_data in self.chatbots.values():
            if bot_data['chatbot_id'] == chatbot_id:
                return bot_data
        return None


storage = InMemoryStorage()

class FastScraper:
    """Fast website scraper"""
   
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        self.timeout = 6
   
    def scrape_page(self, url):
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
            print(f"[Scraper] Error: {e}")
            return None
   
    def scrape_website(self, base_url, progress_callback=None):
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


class SmartAI:
    """AI integration with better error handling"""
   
    def __init__(self):
        self.cache = {}
        self.api_call_count = 0
        self.last_error = None
   
    def call_llm(self, prompt, max_retries=2):
        """Call LLM API with retry logic and detailed error messages"""
        
        # Check API key first
        if not OPENROUTER_API_KEY:
            error_msg = """⚠️ **API Key Not Set!**

Please set your OpenRouter API key:

1. Create a `.env` file in your project folder
2. Add this line: `OPENROUTER_API_KEY=your_key_here`
3. Get your key from: https://openrouter.ai/keys
4. Add credits: https://openrouter.ai/credits (minimum $5)
5. Restart the app

**Or set as environment variable:**
```bash
export OPENROUTER_API_KEY="your_key_here"
```"""
            return error_msg
       
        # Check cache
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:12]
        if cache_key in self.cache:
            print(f"[AI] 💾 Cache hit (call #{self.api_call_count})")
            return self.cache[cache_key]
        
        # Try API call with retries
        for attempt in range(max_retries):
            try:
                self.api_call_count += 1
                print(f"[AI] 🚀 API Call #{self.api_call_count} (Attempt {attempt + 1}/{max_retries})")
                print(f"[AI] 📝 Model: {MODEL}")
                
                response = requests.post(
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
                
                print(f"[AI] 📊 Status Code: {response.status_code}")
                
                # Success
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        answer = data["choices"][0]["message"]["content"].strip()
                        self.cache[cache_key] = answer
                        print(f"[AI] ✅ Success! Response length: {len(answer)} chars")
                        return answer
                    else:
                        print(f"[AI] ⚠️ Unexpected response format: {data}")
                        return "I received an unexpected response. Please try again."
                
                # Authentication Error
                elif response.status_code == 401:
                    error_msg = """❌ **Authentication Failed!**

Your API key is invalid or expired.

**Fix this:**
1. Go to https://openrouter.ai/keys
2. Check if your key is active
3. Generate a new key if needed
4. Update your `.env` file with the new key
5. Restart the app

**Current key preview:** `{}`""".format(OPENROUTER_API_KEY[:15] + "...")
                    self.last_error = error_msg
                    print(f"[AI] ❌ Auth failed: {response.text}")
                    return error_msg
                
                # Insufficient Credits
                elif response.status_code == 402:
                    error_msg = """💰 **Insufficient Credits!**

You need to add credits to your OpenRouter account.

**Fix this:**
1. Go to https://openrouter.ai/credits
2. Click "Add Credits"
3. Add at least $5 (recommended $10)
4. Wait 1-2 minutes for credits to appear
5. Try again

**Why:** Even though some models are "free", OpenRouter requires a minimum balance."""
                    self.last_error = error_msg
                    print(f"[AI] 💰 No credits: {response.text}")
                    return error_msg
                
                # Rate Limit
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"[AI] ⏱️ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    
                    error_msg = """⏱️ **Rate Limit Exceeded!**

The free model has strict rate limits.

**Solutions:**
1. **Wait 30 seconds** and try again
2. **Use a paid model** (much more reliable):
   - Change `CURRENT_MODEL = "cheap"` in code
   - Costs only ~$0.002 per message
3. **Add credits** to increase your rate limits

**Current model:** {}""".format(MODEL)
                    self.last_error = error_msg
                    return error_msg
                
                # Other Errors
                else:
                    error_detail = response.text[:300]
                    error_msg = f"""⚠️ **API Error {response.status_code}**

Something went wrong with the API request.

**Error details:**
```
{error_detail}
```

**Try:**
1. Wait a few seconds and try again
2. Check https://openrouter.ai/activity for more info
3. Try a different model (change CURRENT_MODEL in code)"""
                    self.last_error = error_msg
                    print(f"[AI] ❌ Error {response.status_code}: {error_detail}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    return error_msg
                   
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[AI] ⏱️ Timeout, retrying...")
                    time.sleep(2)
                    continue
                return "⏱️ Request timed out. Please try again."
            
            except requests.exceptions.ConnectionError:
                return """🌐 **Connection Error**

Cannot reach OpenRouter API.

**Check:**
1. Your internet connection
2. Firewall/antivirus isn't blocking requests
3. OpenRouter status: https://status.openrouter.ai
4. Try using a VPN if in a restricted region"""
            
            except Exception as e:
                print(f"[AI] ❌ Unexpected error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"❌ Unexpected error: {type(e).__name__}. Please try again."
        
        return "Failed after multiple attempts. Please check your settings and try again."
    
    def get_stats(self):
        """Get API usage statistics"""
        return {
            "total_calls": self.api_call_count,
            "cache_size": len(self.cache),
            "last_error": self.last_error
        }

class UniversalChatbot:
    """Universal chatbot"""
   
    def __init__(self, company_name, website_url, chatbot_id):
        self.company_name = company_name
        self.website_url = website_url
        self.chatbot_id = chatbot_id
        self.pages = []
        self.contact_info = {}
        self.ready = False
        self.ai = SmartAI()
   
    def initialize(self, progress_callback=None):
        try:
            scraper = FastScraper()
            self.pages, self.contact_info = scraper.scrape_website(self.website_url, progress_callback)
            self.ready = True
            print(f"[Bot] ✅ Initialized: {self.company_name}")
            return True
        except Exception as e:
            print(f"[Bot] ❌ Init error: {e}")
            return False
   
    def ask(self, question):
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
# UTILITY FUNCTIONS
# ========================================
def generate_embed_code(chatbot_id, company_name):
    """Generate HTML embed code"""
    embed_url = f"{DEPLOYMENT_URL}?mode=embed&id={chatbot_id}"
    
    return f'''<!-- {company_name} AI Chatbot -->
<div id="ai-chatbot-{chatbot_id}"></div>
<script>
(function() {{
  var embedUrl = '{embed_url}';
  var btn = document.createElement('button');
  btn.innerHTML = '💬 Chat';
  btn.style.cssText = 'position:fixed;bottom:20px;right:20px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:none;border-radius:50px;padding:15px 30px;font-size:16px;font-weight:600;cursor:pointer;box-shadow:0 4px 15px rgba(102,126,234,0.4);z-index:999999;transition:all 0.3s;font-family:-apple-system,sans-serif;';
  
  var container = document.createElement('div');
  container.style.cssText = 'position:fixed;bottom:90px;right:20px;width:400px;height:600px;max-width:calc(100vw - 40px);max-height:calc(100vh - 120px);border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,0.3);z-index:999998;display:none;background:white;';
  
  var iframe = document.createElement('iframe');
  iframe.src = embedUrl;
  iframe.style.cssText = 'width:100%;height:100%;border:none;border-radius:12px;';
  container.appendChild(iframe);
  
  btn.onclick = function() {{
    var show = container.style.display === 'none';
    container.style.display = show ? 'block' : 'none';
    btn.innerHTML = show ? '✕' : '💬 Chat';
  }};
  
  document.body.appendChild(btn);
  document.body.appendChild(container);
}})();
</script>'''


def validate_email(email):
    if not email or not email.strip():
        return False
    return '@' in email and '.' in email.split('@')[-1]


def init_session():
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
        'embed_chat_history': [],
        'embed_question_count': 0,
        'embed_lead_captured': False,
        'embed_lead_mode': None,
        'embed_lead_data': {}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ========================================
# EMBED MODE
# ========================================
def render_embed_mode():
    st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")
    
    st.markdown("""
    <style>
        .stApp { background: transparent; }
        .main .block-container { padding: 1rem; max-width: 100%; }
        header, footer { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    
    chatbot_id = st.query_params.get("id")
    
    if not chatbot_id:
        st.error("⚠️ No chatbot ID provided")
        return
    
    bot_data = storage.get_chatbot_by_id(chatbot_id)
    
    if not bot_data:
        st.error("⚠️ Chatbot not found.")
        return
    
    if chatbot_id not in st.session_state.chatbots:
        bot = UniversalChatbot(bot_data['company_name'], bot_data['website_url'], chatbot_id)
        with st.spinner("Initializing..."):
            if bot.initialize():
                st.session_state.chatbots[chatbot_id] = bot
            else:
                st.error("Failed to initialize chatbot")
                return
    
    bot = st.session_state.chatbots[chatbot_id]
    
    st.markdown(f"### 💬 {bot.company_name}")
    st.caption("AI-powered assistant")
    
    for msg in st.session_state.embed_chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Lead capture (simplified for embed mode)
    if st.session_state.embed_lead_mode and not st.session_state.embed_lead_captured:
        st.info("Please share your contact info to continue")
        email = st.text_input("Email", key="embed_email")
        if st.button("Submit"):
            if validate_email(email):
                storage.save_lead(chatbot_id, bot.company_name, "Visitor", email,
                                "Not provided", st.session_state.session_id,
                                st.session_state.embed_question_count,
                                st.session_state.embed_chat_history)
                st.session_state.embed_lead_captured = True
                st.session_state.embed_lead_mode = None
                st.success("✅ Thank you!")
                st.rerun()
    
    if question := st.chat_input("Ask anything..."):
        st.session_state.embed_chat_history.append({"role": "user", "content": question})
        
        with st.spinner("Thinking..."):
            answer = bot.ask(question)
        
        st.session_state.embed_chat_history.append({"role": "assistant", "content": answer})
        st.session_state.embed_question_count += 1
        
        if st.session_state.embed_question_count >= 3 and not st.session_state.embed_lead_captured:
            st.session_state.embed_lead_mode = 'ask_email'
        
        st.rerun()


# ========================================
# MAIN APPLICATION
# ========================================
def main():
    mode = st.query_params.get("mode")
    if mode == "embed":
        render_embed_mode()
        return
    
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    init_session()
   
    st.title("🤖 AI Chatbot Lead Generator")
    st.caption("Create • Deploy • Capture Leads")
   
    # API Key Check with detailed help
    if not OPENROUTER_API_KEY:
        st.error("❌ **API Key Not Set!**")
        
        with st.expander("🔧 **How to Fix - Click Here**", expanded=True):
            st.markdown("""
            ### Quick Setup (2 minutes):
            
            **Step 1: Get API Key**
            1. Go to [OpenRouter Keys](https://openrouter.ai/keys)
            2. Sign up or login
            3. Click "Create Key"
            4. Copy your key (starts with `sk-or-v1-`)
            
            **Step 2: Set the Key**
            
            Create a file named `.env` in your project folder:
            ```
            OPENROUTER_API_KEY=sk-or-v1-your-key-here
            ```
            
            **Step 3: Add Credits**
            1. Go to [OpenRouter Credits](https://openrouter.ai/credits)
            2. Add at least $5
            3. Wait 1-2 minutes
            
            **Step 4: Restart App**
            ```bash
            streamlit run app.py
            ```
            
            **Alternative: Set Environment Variable**
            ```bash
            # On Windows PowerShell:
            $env:OPENROUTER_API_KEY="your_key_here"
            streamlit run app.py
            
            # On macOS/Linux:
            export OPENROUTER_API_KEY="your_key_here"
            streamlit run app.py
            ```
            """)
        st.stop()
    
    # Sidebar
    st.sidebar.title("🏢 Admin Panel")
    
    # API Status with test button
    with st.sidebar.expander("🔑 API Status", expanded=True):
        key_preview = f"{OPENROUTER_API_KEY[:15]}...{OPENROUTER_API_KEY[-8:]}"
        st.success(f"✅ Key: {key_preview}")
        st.info(f"🤖 Model: {MODEL}")
        st.caption(f"🌐 URL: {DEPLOYMENT_URL}")
        
        if st.button("🧪 Test API", type="primary"):
            with st.spinner("Testing API connection..."):
                test_ai = SmartAI()
                result = test_ai.call_llm("Say 'API working!' in 3 words")
                
                if "⚠️" in result or "❌" in result or "💰" in result:
                    st.error("API Test Failed")
                    st.markdown(result)
                else:
                    st.success("✅ API Working!")
                    st.write(f"Response: {result}")
    
    # Create Chatbot
    with st.sidebar.expander("➕ Create Chatbot", expanded=True):
        name = st.text_input("Company Name")
        url = st.text_input("Website URL")
        
        if st.button("🚀 Create", type="primary"):
            if name and url:
                chatbot_id = hashlib.md5(f"{name}{url}{time.time()}".encode()).hexdigest()[:12]
                slug = re.sub(r'[^a-z0-9]+', '-', name.lower())
                
                progress = st.progress(0)
                status = st.empty()
                
                def cb(done, total, url_str):
                    progress.progress(done/total)
                    status.text(f"Scraping {done}/{total}...")
                
                bot = UniversalChatbot(name, url, chatbot_id)
                if bot.initialize(cb):
                    st.session_state.chatbots[slug] = bot
                    st.session_state.current_company = slug
                    st.session_state.chat_history = []
                    st.session_state.question_count = 0
                    st.session_state.lead_captured = False
                    
                    embed = generate_embed_code(chatbot_id, name)
                    storage.save_chatbot(chatbot_id, name, url, embed)
                    st.success("✅ Created!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize")
            else:
                st.warning("Please enter both fields")
    
    # List Chatbots
    if st.session_state.chatbots:
        st.sidebar.subheader("📋 Chatbots")
        for slug, bot in st.session_state.chatbots.items():
            col1, col2 = st.sidebar.columns([3,1])
            with col1:
                if st.button(f"💬 {bot.company_name}", key=f"sel_{slug}"):
                    st.session_state.current_company = slug
                    st.session_state.chat_history = []
                    st.session_state.question_count = 0
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{slug}"):
                    del st.session_state.chatbots[slug]
                    if st.session_state.current_company == slug:
                        st.session_state.current_company = None
                    st.rerun()
    
    if st.sidebar.button("📊 View Leads"):
        st.subheader("📊 Captured Leads")
        leads = storage.get_leads()
        if leads:
            for lead in leads:
                with st.expander(f"🎯 {lead['username']} - {lead['company_name']}"):
                    st.write(f"**Email:** {lead['mailid']}")
                    st.write(f"**Phone:** {lead['phonenumber']}")
                    st.write(f"**Questions:** {lead['questions_asked']}")
                    st.write(f"**Time:** {lead['timestart']}")
        else:
            st.info("No leads yet")
        return
    
    # Main Interface
    if not st.session_state.current_company:
        st.info("👈 Create a chatbot to start!")
        
        st.markdown("### 📖 Quick Start Guide")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1️⃣ Create\n- Enter company name\n- Add website URL\n- Let AI learn")
        with col2:
            st.markdown("#### 2️⃣ Test\n- Chat with bot\n- Verify responses\n- Check lead capture")
        with col3:
            st.markdown("#### 3️⃣ Deploy\n- Get embed code\n- Paste on website\n- Start capturing!")
        return
    
    # Active Chatbot Interface
    bot = st.session_state.chatbots[st.session_state.current_company]
    
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.subheader(f"💬 {bot.company_name}")
    with col2:
        st.metric("Questions", st.session_state.question_count)
    with col3:
        if st.session_state.lead_captured:
            st.success("✅ Lead")
        else:
            st.info("🎯 Pending")
    
    # Get Embed Code
    with st.expander("🚀 Deploy to Website", expanded=False):
        st.markdown("### Copy & Paste This Code")
        code = generate_embed_code(bot.chatbot_id, bot.company_name)
        st.code(code, language='html')
        st.download_button("📥 Download", code, f"{bot.company_name}_widget.html", "text/html")
        
        st.info("""
        **Where to paste:**
        - Before `</body>` tag in your HTML
        - In WordPress Custom HTML widget
        - In Shopify theme.liquid file
        - See full deployment guide for more platforms
        """)
    
    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Lead Capture Form
    if st.session_state.lead_capture_mode and not st.session_state.lead_captured:
        st.markdown("---")
        st.markdown("### 🎯 Let's Connect!")
        
        if st.session_state.lead_capture_mode == 'ask_name':
            name = st.text_input("Your Name", key="name_input", placeholder="John Doe")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Continue", type="primary", key="submit_name"):
                    if name and name.strip():
                        st.session_state.lead_data['name'] = name.strip()
                        st.session_state.lead_capture_mode = 'ask_email'
                        st.rerun()
            with col2:
                if st.button("⏭️ Skip", key="skip_name"):
                    st.session_state.lead_data['name'] = "Anonymous"
                    st.session_state.lead_capture_mode = 'ask_email'
                    st.rerun()
        
        elif st.session_state.lead_capture_mode == 'ask_email':
            email = st.text_input("Email", key="email_input", placeholder="john@example.com")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Continue", type="primary", key="submit_email"):
                    if email and validate_email(email):
                        st.session_state.lead_data['email'] = email.strip()
                        st.session_state.lead_capture_mode = 'ask_phone'
                        st.rerun()
            with col2:
                if st.button("⏭️ Skip", key="skip_email"):
                    st.session_state.lead_data['email'] = "not_provided@example.com"
                    st.session_state.lead_capture_mode = 'ask_phone'
                    st.rerun()
        
        elif st.session_state.lead_capture_mode == 'ask_phone':
            phone = st.text_input("Phone (Optional)", key="phone_input", placeholder="+1 234 567 8900")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Submit", type="primary", key="submit_phone"):
                    phone_value = phone.strip() if phone else "Not provided"
                    st.session_state.lead_data['phone'] = phone_value
                    
                    with st.spinner("Saving..."):
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
                        st.success("✅ Thank you!")
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
                        st.success("✅ Thank you!")
                        time.sleep(1)
                        st.rerun()
    
    # Chat Input
    if question := st.chat_input("Ask anything...",
                                  disabled=bool(st.session_state.lead_capture_mode)):
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.spinner("Thinking..."):
            answer = bot.ask(question)
        
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
