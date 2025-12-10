import streamlit as st

st.set_page_config(page_title="Code Analysis - Universal AI Chatbot", page_icon="🔍", layout="wide")

# Sidebar navigation
st.sidebar.title("📋 Navigation")
section = st.sidebar.radio(
    "Choose Section:",
    ["Overview", "Architecture", "Persistence System", "Sharing System", "Critical Issues", "Improvements"]
)

st.title("🔍 Deep Code Analysis: Universal AI Chatbot")
st.markdown("---")

# OVERVIEW SECTION
if section == "Overview":
    st.header("🌐 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What It Does")
        st.info("""
        A **Universal AI Chatbot Generator** that creates custom chatbots for any company by:
        - 🕷️ Scraping their website automatically
        - 🤖 Building a knowledge base from content
        - 💬 Answering questions using AI (RAG approach)
        - 🔗 Creating shareable public links
        - 📦 Generating embed codes for websites
        """)
        
    with col2:
        st.subheader("Tech Stack")
        st.success("""
        - **Frontend**: Streamlit
        - **Web Scraping**: BeautifulSoup + ThreadPoolExecutor
        - **AI Model**: OpenRouter API (kat-coder-pro)
        - **Storage**: Streamlit Session State
        - **Pattern Matching**: Regex for contact extraction
        """)
    
    st.subheader("🎯 Key Features")
    
    features = {
        "Web Scraping": {
            "icon": "🕷️",
            "desc": "Parallel scraping of up to 50 pages per website",
            "tech": "ThreadPoolExecutor with 10 workers"
        },
        "RAG System": {
            "icon": "🧠",
            "desc": "Retrieval Augmented Generation for contextual answers",
            "tech": "Word overlap scoring + LLM prompting"
        },
        "Contact Extraction": {
            "icon": "📞",
            "desc": "Automatic extraction of emails and phone numbers",
            "tech": "Regex patterns with validation"
        },
        "Multi-Tenancy": {
            "icon": "🏢",
            "desc": "Manage multiple company chatbots in one app",
            "tech": "Session state with slug-based routing"
        }
    }
    
    for feature, info in features.items():
        with st.expander(f"{info['icon']} {feature}"):
            st.write(f"**Description**: {info['desc']}")
            st.code(info['tech'], language="text")

# ARCHITECTURE SECTION
elif section == "Architecture":
    st.header("🏗️ System Architecture")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Class Structure", "Data Flow", "Scraping Logic", "AI Integration"])
    
    with tab1:
        st.subheader("Class Hierarchy")
        
        st.code("""
# 1. FastScraper - Web scraping engine
class FastScraper:
    - clean_text()          # Text normalization
    - extract_contact_info() # Email/phone extraction
    - scrape_page()         # Single page scraper
    - get_urls_to_scrape()  # URL discovery
    - scrape_website()      # Parallel orchestration

# 2. SmartAI - LLM wrapper
class SmartAI:
    - response_cache        # In-memory cache
    - call_llm()           # API call with retry

# 3. UniversalChatbot - Main bot logic
class UniversalChatbot:
    - initialize()         # Scrape and prepare
    - get_context()        # RAG retrieval
    - ask()               # Answer questions
    - to_dict()           # Serialization
    - from_dict()         # Deserialization
        """, language="python")
        
        st.info("💡 **Design Pattern**: The code uses a layered architecture separating concerns - scraping, AI, and chatbot logic.")
    
    with tab2:
        st.subheader("Request Flow Diagram")
        
        st.code("""
User Question
    ↓
Check if greeting/contact query → Handle directly
    ↓
If not → get_context() → Rank pages by word overlap
    ↓
Select top 5 pages → Extract content (max 1000 chars each)
    ↓
Build prompt with context + question
    ↓
call_llm() → Check cache → If miss, call OpenRouter API
    ↓
Return answer to user
        """, language="text")
        
        st.warning("⚠️ **Bottleneck**: Every question triggers context retrieval + LLM call (no conversation memory)")
    
    with tab3:
        st.subheader("Web Scraping Pipeline")
        
        st.code("""
# Step 1: URL Discovery
get_urls_to_scrape(base_url):
    1. Generate predefined paths (/about, /contact, etc.)
    2. Fetch homepage
    3. Extract all internal links (up to 60)
    4. Filter same-domain links
    5. Return up to 50 URLs

# Step 2: Parallel Scraping
scrape_website():
    ThreadPoolExecutor(max_workers=10):
        For each URL:
            - scrape_page()
            - Extract title
            - Remove nav/footer/scripts
            - Get main content or fallback to h1/h2/p
            - Clean and deduplicate lines
            - Extract contact info
    
# Step 3: Content Processing
    - Keep only unique lines (>25 chars)
    - Limit to 50 lines per page
    - Total content cap: 4000 chars/page
        """, language="python")
    
    with tab4:
        st.subheader("AI Integration Details")
        
        st.code("""
# OpenRouter API Call
call_llm(prompt):
    1. Generate MD5 cache key from prompt
    2. Check response_cache (in-memory dict)
    3. If cache miss:
        - POST to https://openrouter.ai/api/v1/chat/completions
        - Model: "kwaipilot/kat-coder-pro:free"
        - Temperature: 0.7
        - Max tokens: 400
        - Timeout: 45 seconds
    4. Retry once if fails (2-second delay)
    5. Cache successful responses
    6. Return content or error message

# Prompt Structure
"You are a helpful AI assistant for {company_name}.

Based on the following information from their website, 
answer the user's question clearly.

COMPANY INFORMATION:
{context[:2500]}

USER QUESTION: {question}

Instructions:
- Provide a helpful answer in 2-4 sentences
- Be specific and use details from the context
- Be friendly and professional"
        """, language="python")

# PERSISTENCE SECTION
elif section == "Persistence System":
    st.header("💾 Persistence & State Management")
    
    st.subheader("Current Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ What Works")
        st.success("""
        **Session State Storage**:
        - Chatbots stored in `st.session_state.chatbots`
        - Dictionary keyed by slug
        - Serialization via `to_dict()` / `from_dict()`
        - Survives page reruns within same session
        """)
        
        st.code("""
# Storage functions
def save_chatbots():
    chatbots_data = {}
    for slug, bot in st.session_state.chatbots.items():
        chatbots_data[slug] = bot.to_dict()
    st.session_state.chatbots_data = chatbots_data

def load_chatbots():
    if 'chatbots_data' in st.session_state:
        st.session_state.chatbots = {}
        for slug, data in st.session_state.chatbots_data.items():
            st.session_state.chatbots[slug] = UniversalChatbot.from_dict(data)
        """, language="python")
    
    with col2:
        st.markdown("### ❌ Critical Limitations")
        st.error("""
        **Session-Only Persistence**:
        - Data lost when browser closes
        - Not shared across users
        - No real database
        - Public links won't work for others
        - Embed codes are non-functional
        """)
        
        st.warning("""
        **Why This Fails**:
        1. User A creates chatbot → Stored in User A's session
        2. User B clicks shared link → New session, no data
        3. Result: 404 or "chatbot not found"
        """)
    
    st.markdown("---")
    st.subheader("🔧 Required Improvements")
    
    tabs = st.tabs(["Database Schema", "Implementation Options", "Code Changes"])
    
    with tabs[0]:
        st.code("""
# Recommended Database Schema

TABLE: chatbots
- id (UUID, primary key)
- slug (string, unique index)
- company_name (string)
- website_url (string)
- created_at (timestamp)
- updated_at (timestamp)
- status (enum: 'scraping', 'ready', 'failed')
- error_message (text, nullable)
- contact_emails (json array)
- contact_phones (json array)

TABLE: chatbot_pages
- id (UUID, primary key)
- chatbot_id (UUID, foreign key)
- url (string)
- title (string)
- content (text, max 4000 chars)
- scraped_at (timestamp)

TABLE: chat_messages
- id (UUID, primary key)
- chatbot_id (UUID, foreign key)
- session_id (string)
- role (enum: 'user', 'assistant')
- message (text)
- created_at (timestamp)

INDEXES:
- chatbots.slug (unique, for fast lookups)
- chatbot_pages.chatbot_id (for joins)
- chat_messages.chatbot_id + session_id (for history)
        """, language="sql")
    
    with tabs[1]:
        st.markdown("### Database Options")
        
        options = {
            "SQLite + Streamlit": {
                "pros": ["Simple", "No external deps", "File-based"],
                "cons": ["Not multi-user safe", "Limited scalability"],
                "code": "import sqlite3; conn = sqlite3.connect('chatbots.db')"
            },
            "PostgreSQL": {
                "pros": ["Production-ready", "Multi-user", "ACID compliant"],
                "cons": ["Requires hosting", "More complex setup"],
                "code": "import psycopg2; conn = psycopg2.connect(DATABASE_URL)"
            },
            "Supabase": {
                "pros": ["Free tier", "Realtime", "Built-in auth", "Hosted"],
                "cons": ["Vendor lock-in", "External dependency"],
                "code": "from supabase import create_client; supabase = create_client(url, key)"
            },
            "MongoDB": {
                "pros": ["Schema-less", "JSON-native", "Easy for this use case"],
                "cons": ["NoSQL complexity", "Requires hosting"],
                "code": "from pymongo import MongoClient; client = MongoClient(MONGO_URI)"
            }
        }
        
        for db, info in options.items():
            with st.expander(f"🗄️ {db}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pros:**")
                    for pro in info['pros']:
                        st.markdown(f"✅ {pro}")
                with col2:
                    st.markdown("**Cons:**")
                    for con in info['cons']:
                        st.markdown(f"❌ {con}")
                st.code(info['code'], language="python")
    
    with tabs[2]:
        st.markdown("### Code Changes Needed")
        
        st.code("""
# 1. Database connection
import psycopg2
from psycopg2.extras import Json

DATABASE_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)

# 2. Save chatbot to database
def save_chatbot_to_db(chatbot):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chatbots (slug, company_name, website_url, 
                             contact_emails, contact_phones, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (slug) DO UPDATE SET
            updated_at = NOW(),
            contact_emails = EXCLUDED.contact_emails,
            contact_phones = EXCLUDED.contact_phones,
            status = EXCLUDED.status
    ''', (
        chatbot.slug,
        chatbot.company_name,
        chatbot.website_url,
        Json(chatbot.contact_info['emails']),
        Json(chatbot.contact_info['phones']),
        'ready' if chatbot.ready else 'failed'
    ))
    
    chatbot_id = cursor.fetchone()[0]
    
    # Save pages
    for page in chatbot.pages:
        cursor.execute('''
            INSERT INTO chatbot_pages (chatbot_id, url, title, content)
            VALUES (%s, %s, %s, %s)
        ''', (chatbot_id, page['url'], page['title'], page['content']))
    
    conn.commit()

# 3. Load chatbot from database
def load_chatbot_from_db(slug):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM chatbots WHERE slug = %s', (slug,))
    row = cursor.fetchone()
    
    if not row:
        return None
    
    chatbot = UniversalChatbot(row['company_name'], row['website_url'], slug)
    chatbot.contact_info = {
        'emails': row['contact_emails'],
        'phones': row['contact_phones']
    }
    chatbot.ready = (row['status'] == 'ready')
    
    # Load pages
    cursor.execute('SELECT * FROM chatbot_pages WHERE chatbot_id = %s', (row['id'],))
    chatbot.pages = [
        {'url': p['url'], 'title': p['title'], 'content': p['content']}
        for p in cursor.fetchall()
    ]
    
    return chatbot

# 4. Update main() to load from DB on shareable link access
def main():
    query_params = st.query_params
    bot_slug = query_params.get("bot", None)
    
    if bot_slug:
        # Load from database instead of session state
        chatbot = load_chatbot_from_db(bot_slug)
        if chatbot:
            st.session_state.chatbots[bot_slug] = chatbot
            st.session_state.current_company = bot_slug
        else:
            st.error("Chatbot not found")
            return
        """, language="python")

# SHARING SYSTEM
elif section == "Sharing System":
    st.header("🔗 Sharing & Embedding System")
    
    st.subheader("How It Currently Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Shareable Links")
        st.code("""
def generate_shareable_link(slug: str) -> str:
    base_url = "https://your-app-domain.streamlit.app"
    return f"{base_url}?bot={slug}"

# Example output:
# https://your-app-domain.streamlit.app?bot=acme-corp
        """, language="python")
        
        st.info("""
        **URL Parameter Handling**:
        ```python
        query_params = st.query_params
        bot_slug = query_params.get("bot", None)
        
        if bot_slug and bot_slug in st.session_state.chatbots:
            st.session_state.current_company = bot_slug
        ```
        """)
    
    with col2:
        st.markdown("### Embed Codes")
        st.code("""
def generate_embed_code(slug: str, company_name: str) -> str:
    base_url = "https://your-app-domain.streamlit.app"
    
    embed_code = f'''
    <!-- {company_name} AI Chatbot -->
    <div id="ai-chatbot-{slug}"></div>
    <script>
      var iframe = document.createElement('iframe');
      iframe.src = '{base_url}?bot={slug}&embed=true';
      iframe.style.width = '400px';
      iframe.style.height = '600px';
      iframe.style.border = 'none';
      document.getElementById('ai-chatbot-{slug}')
              .appendChild(iframe);
    </script>
    '''
    return embed_code
        """, language="python")
    
    st.markdown("---")
    st.subheader("🎨 Embed Mode Features")
    
    st.code("""
# Detect embed mode from URL parameter
is_embed = query_params.get("embed", "false") == "true"

# Hide sidebar and adjust layout for embedding
if is_embed:
    st.markdown('''
    <style>
    [data-testid="stSidebar"] { display: none; }
    .main { padding: 1rem; }
    </style>
    ''', unsafe_allow_html=True)

# Simpler interface without management features
if not is_embed:
    # Show full dashboard
else:
    # Show only chat interface
    """, language="python")
    
    st.markdown("---")
    st.subheader("❌ Critical Problems")
    
    st.error("""
    **Why This System Doesn't Work**:
    
    1. **Session State Limitation**: 
       - Chatbots only exist in creator's session
       - Other users can't access via shared link
       
    2. **No Cross-User Access**:
       - Each Streamlit session is isolated
       - No shared data store
       
    3. **Embed Codes are Non-Functional**:
       - External websites load new Streamlit session
       - Chatbot data not available in that session
    """)
    
    st.markdown("---")
    st.subheader("✅ Complete Solution")
    
    solution_tabs = st.tabs(["Architecture Fix", "Implementation", "Security"])
    
    with solution_tabs[0]:
        st.markdown("### Required Architecture")
        
        st.code("""
┌─────────────────────────────────────────────────────┐
│                  User's Browser                     │
│  Clicks: https://app.com?bot=acme-corp             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit App Server                   │
│  1. Parse ?bot=acme-corp from URL                  │
│  2. Query database for slug='acme-corp'            │
│  3. Load chatbot data (pages, contacts, etc.)      │
│  4. Render chat interface with data                │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                    Database                         │
│  PostgreSQL / MongoDB / Supabase                   │
│  - chatbots table (permanent storage)              │
│  - chatbot_pages table (scraped content)           │
│  - Indexed by slug for fast lookups                │
└─────────────────────────────────────────────────────┘
        """, language="text")
    
    with solution_tabs[1]:
        st.markdown("### Full Implementation")
        
        st.code("""
# Add to your code:

import os
import psycopg2
from psycopg2.extras import Json, RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")

class ChatbotDB:
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
    
    def save_chatbot(self, chatbot):
        '''Save chatbot to permanent storage'''
        cursor = self.conn.cursor()
        
        try:
            # Insert or update chatbot
            cursor.execute('''
                INSERT INTO chatbots 
                (slug, company_name, website_url, contact_emails, 
                 contact_phones, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (slug) 
                DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    website_url = EXCLUDED.website_url,
                    contact_emails = EXCLUDED.contact_emails,
                    contact_phones = EXCLUDED.contact_phones,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING id
            ''', (
                chatbot.slug,
                chatbot.company_name,
                chatbot.website_url,
                Json(chatbot.contact_info['emails']),
                Json(chatbot.contact_info['phones']),
                'ready' if chatbot.ready else 'failed'
            ))
            
            chatbot_id = cursor.fetchone()[0]
            
            # Delete old pages
            cursor.execute('DELETE FROM chatbot_pages WHERE chatbot_id = %s', 
                         (chatbot_id,))
            
            # Insert new pages
            for page in chatbot.pages:
                cursor.execute('''
                    INSERT INTO chatbot_pages 
                    (chatbot_id, url, title, content, scraped_at)
                    VALUES (%s, %s, %s, %s, NOW())
                ''', (chatbot_id, page['url'], page['title'], page['content']))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error saving chatbot: {e}")
            return False
    
    def load_chatbot(self, slug):
        '''Load chatbot from permanent storage'''
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Get chatbot metadata
        cursor.execute('SELECT * FROM chatbots WHERE slug = %s', (slug,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Create chatbot object
        chatbot = UniversalChatbot(
            row['company_name'], 
            row['website_url'], 
            slug
        )
        
        chatbot.contact_info = {
            'emails': row['contact_emails'] or [],
            'phones': row['contact_phones'] or []
        }
        chatbot.ready = (row['status'] == 'ready')
        chatbot.created_at = row['created_at'].isoformat()
        
        # Load pages
        cursor.execute('''
            SELECT url, title, content 
            FROM chatbot_pages 
            WHERE chatbot_id = %s
        ''', (row['id'],))
        
        chatbot.pages = [dict(p) for p in cursor.fetchall()]
        
        return chatbot

# Update main() function
def main():
    st.set_page_config(...)
    
    init_session()
    db = ChatbotDB()
    
    # Check for URL parameters
    query_params = st.query_params
    bot_slug = query_params.get("bot", None)
    is_embed = query_params.get("embed", "false") == "true"
    
    # If accessing via shareable link
    if bot_slug:
        # Try loading from database
        chatbot = db.load_chatbot(bot_slug)
        
        if chatbot:
            # Store in session for this user
            st.session_state.chatbots[bot_slug] = chatbot
            st.session_state.current_company = bot_slug
        else:
            st.error("❌ Chatbot not found")
            st.info("This chatbot may have been deleted or the link is invalid.")
            return
    
    # ... rest of your code
    
    # When creating new chatbot
    if st.button("🚀 Create Chatbot"):
        chatbot = UniversalChatbot(company_name, website_url, slug)
        success = chatbot.initialize(callback)
        
        if success:
            # Save to database (not just session)
            if db.save_chatbot(chatbot):
                st.session_state.chatbots[slug] = chatbot
                st.success("✅ Chatbot created and saved!")
            else:
                st.error("❌ Failed to save chatbot")
        """, language="python")
    
    with solution_tabs[2]:
        st.markdown("### Security Considerations")
        
        st.warning("""
        **⚠️ Current Security Issues**:
        - No authentication
        - Anyone can create unlimited chatbots
        - No rate limiting on scraping
        - API keys exposed in environment
        - No input validation on URLs
        """)
        
        st.code("""
# Recommended security additions:

# 1. Rate limiting
from functools import wraps
import time

def rate_limit(max_calls=5, period=3600):
    '''Limit chatbot creation to 5 per hour per user'''
    calls = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = get_user_id()  # Use session ID or IP
            now = time.time()
            
            if user_id not in calls:
                calls[user_id] = []
            
            # Remove old calls
            calls[user_id] = [t for t in calls[user_id] if now - t < period]
            
            if len(calls[user_id]) >= max_calls:
                raise Exception("Rate limit exceeded")
            
            calls[user_id].append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# 2. URL validation
def validate_url(url):
    '''Validate URL before scraping'''
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL scheme")
    
    # Block internal IPs
    parsed = urlparse(url)
    if parsed.hostname in ['localhost', '127.0.0.1'] or \
       parsed.hostname.startswith('192.168.') or \
       parsed.hostname.startswith('10.'):
        raise ValueError("Internal URLs not allowed")
    
    return True

# 3. User authentication (optional)
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'chatbot_app',
    'secret_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
    main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
        """, language="python")

# CRITICAL ISSUES
elif section == "Critical Issues":
    st.header("🚨 Critical Issues")
    
    st.error("### Major Problems That Break Functionality")
    
    issues = [
        {
            "severity": "critical",
            "title": "Session-Only Persistence",
            "description": "Chatbots are stored ONLY in st.session_state, which is user-specific and session-specific. When a user closes their browser or another user tries to access a shared link, the data is gone.",
            "impact": "Shared links don't work for anyone except the creator, and only during their active session."
        },
        {
            "severity": "critical",
            "title": "Non-Functional Embed Codes",
            "description": "Embed codes create iframes that load the Streamlit app in a new session. Since chatbot data isn't in a database, the embedded chatbot shows 'not found'.",
            "impact": "External websites cannot embed these chatbots. The feature is completely broken."
        },
        {
            "severity": "high",
            "title": "No User Authentication",
            "description": "Anyone can create unlimited chatbots, scrape any website, and consume API credits. No user management or access control.",
            "impact": "Abuse potential, API cost explosion, no accountability."
        },
        {
            "severity": "high",
            "title": "Missing Rate Limiting",
            "description": "No limits on chatbot creation or API calls. A user could create 1000 chatbots and scrape 50,000 pages.",
            "impact": "Service abuse, potential legal issues with aggressive scraping, cost overruns."
        },
        {
            "severity": "medium",
            "title": "No Conversation Memory",
            "description": "Each question is answered independently. The AI doesn't remember previous messages in the conversation.",
            "impact": "Poor user experience. Can't have contextual follow-up questions."
        },
        {
            "severity": "medium",
            "title": "Inefficient Context Retrieval",
            "description": "Uses simple word overlap for RAG. No semantic search, no embeddings, no vector database.",
            "impact": "Answers may miss relevant context, especially for paraphrased questions."
        },
        {
            "severity": "medium",
            "title": "Hard-Coded Placeholder URLs",
            "description": "Shareable links use 'https://your-app-domain.streamlit.app' which needs manual replacement.",
            "impact": "Generated links are broken until manually updated in code."
        },
        {
            "severity": "low",
            "title": "No Error Recovery",
            "description": "If scraping fails midway, all data is lost. No partial saves or resume capability.",
            "impact": "Wasted time and API calls when scraping large websites."
        }
    ]
    
    for issue in issues:
        severity_colors = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🔵"
        }
        
        with st.expander(f"{severity_colors[issue['severity']]} [{issue['severity'].upper()}] {issue['title']}"):
            st.markdown(f"**Description**: {issue['description']}")
            st.markdown(f"**Impact**: {issue['impact']}")

# IMPROVEMENTS
elif section == "Improvements":
    st.header("✨ Recommended Improvements")
    
    improvement_tabs = st.tabs(["Priority 1", "Priority 2", "Priority 3", "Nice-to-Have"])
    
    with improvement_tabs[0]:
        st.subheader("🔥 Priority 1: Must Fix Now")
        
        st.markdown("### 1. Add Database Persistence")
        st.code("""
# Use PostgreSQL or Supabase
# See "Persistence System" section for full code

pip install psycopg2-binary
# or
pip install supabase

# Then update save/load functions to use database
        """, language="bash")
        
        st.markdown("### 2. Fix Shareable Links")
        st.code("""
# Replace hard-coded URL with environment variable
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "http://localhost:8501")

def generate_shareable_link(slug: str) -> str:
    return f"{STREAMLIT_APP_URL}?bot={slug}"

def generate_embed_code(slug: str, company_name: str) -> str:
    embed_code = f'''<!-- {company_name} AI Chatbot -->
<div id="ai-chatbot-{slug}"></div>
<script>
  (function() {{
    var iframe = document.createElement('iframe');
    iframe.src = '{STREAMLIT_APP_URL}?bot={slug}&embed=true';
    iframe.style.width = '400px';
    iframe.style.height = '600px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '10px';
    document.getElementById('ai-chatbot-{slug}').appendChild(iframe);
  }})();
</script>'''
    return embed_code
        """, language="python")
        
        st.markdown("### 3. Add Error Handling")
        st.code("""
# Wrap database operations in try-catch
def safe_db_operation(func):
    try:
        return func()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None
        """, language="python")
    
    with improvement_tabs[1]:
        st.subheader("⚡ Priority 2: Enhance Functionality")
        
        st.markdown("### 1. Add Conversation Memory")
        st.code("""
# Store conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def ask_with_memory(self, question: str) -> str:
    # Build context from conversation history
    history_context = "\\n".join([
        f"User: {msg['user']}\\nAssistant: {msg['assistant']}"
        for msg in st.session_state.conversation_history[-3:]  # Last 3 exchanges
    ])
    
    # Add to prompt
    prompt = f'''You are a helpful AI assistant for {self.company_name}.

Previous conversation:
{history_context}

Current context from website:
{context}

USER QUESTION: {question}

Answer based on context and conversation history:'''
    
    answer = self.ai.call_llm(prompt)
    
    # Store in history
    st.session_state.conversation_history.append({
        'user': question,
        'assistant': answer
    })
    
    return answer
        """, language="python")
        
        st.markdown("### 2. Improve RAG with Semantic Search")
        st.code("""
# Use sentence-transformers for better context retrieval
from sentence_transformers import SentenceTransformer
import numpy as np

class ImprovedChatbot(UniversalChatbot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.page_embeddings = []
    
    def initialize(self, progress_callback=None):
        success = super().initialize(progress_callback)
        if success:
            # Generate embeddings for all pages
            texts = [p['content'] for p in self.pages]
            self.page_embeddings = self.model.encode(texts)
        return success
    
    def get_context(self, question: str) -> str:
        # Encode question
        question_embedding = self.model.encode([question])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.page_embeddings, question_embedding)
        
        # Get top 5 most similar pages
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        context_parts = []
        for idx in top_indices:
            context_parts.append(self.pages[idx]['content'][:1000])
        
        return "\\n\\n---\\n\\n".join(context_parts)
        """, language="python")
        
        st.markdown("### 3. Add Analytics & Logging")
        st.code("""
# Track usage statistics
def log_interaction(chatbot_slug, question, answer, response_time):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analytics 
        (chatbot_slug, question, answer, response_time, timestamp)
        VALUES (%s, %s, %s, %s, NOW())
    ''', (chatbot_slug, question, answer, response_time))
    conn.commit()

# Add analytics dashboard
def show_analytics(slug):
    cursor = conn.cursor()
    
    # Total questions
    cursor.execute('''
        SELECT COUNT(*) FROM analytics WHERE chatbot_slug = %s
    ''', (slug,))
    total = cursor.fetchone()[0]
    
    # Average response time
    cursor.execute('''
        SELECT AVG(response_time) FROM analytics WHERE chatbot_slug = %s
    ''', (slug,))
    avg_time = cursor.fetchone()[0]
    
    # Most common questions
    cursor.execute('''
        SELECT question, COUNT(*) as count 
        FROM analytics 
        WHERE chatbot_slug = %s 
        GROUP BY question 
        ORDER BY count DESC 
        LIMIT 10
    ''', (slug,))
    common_questions = cursor.fetchall()
    
    st.metric("Total Questions", total)
    st.metric("Avg Response Time", f"{avg_time:.2f}s")
    st.table(common_questions)
        """, language="python")
    
    with improvement_tabs[2]:
        st.subheader("🚀 Priority 3: Performance & UX")
        
        st.markdown("### 1. Add Caching")
        st.code("""
# Cache scraped data to avoid re-scraping
@st.cache_data(ttl=3600)  # Cache for 1 hour
def scrape_website_cached(url):
    scraper = FastScraper()
    return scraper.scrape_website(url)

# Cache LLM responses at app level
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_cached_response(chatbot_slug, question):
    # Check if we've answered this exact question before
    cursor = conn.cursor()
    cursor.execute('''
        SELECT answer FROM cached_responses 
        WHERE chatbot_slug = %s AND question = %s
        AND created_at > NOW() - INTERVAL '24 hours'
    ''', (chatbot_slug, question))
    
    result = cursor.fetchone()
    if result:
        return result[0]
    return None
        """, language="python")
        
        st.markdown("### 2. Add Loading States")
        st.code("""
# Better UX with progress indicators
def create_chatbot_with_progress(company_name, website_url):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Discovering URLs...")
    progress_bar.progress(10)
    
    chatbot = UniversalChatbot(company_name, website_url)
    
    def callback(done, total, url):
        percentage = int((done / total) * 80) + 10
        progress_bar.progress(percentage)
        status_text.text(f"📄 Scraping {done}/{total}: {url[:50]}...")
    
    success = chatbot.initialize(callback)
    
    if success:
        progress_bar.progress(90)
        status_text.text("💾 Saving to database...")
        db.save_chatbot(chatbot)
        
        progress_bar.progress(100)
        status_text.text("✅ Done!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    return chatbot, success
        """, language="python")
        
        st.markdown("### 3. Add Bulk Operations")
        st.code("""
# Allow users to create multiple chatbots at once
def bulk_create_chatbots(urls_file):
    urls = urls_file.read().decode('utf-8').splitlines()
    
    results = []
    for i, url in enumerate(urls):
        st.write(f"Processing {i+1}/{len(urls)}: {url}")
        
        try:
            company_name = url.split('//')[1].split('.')[0].title()
            chatbot = UniversalChatbot(company_name, url)
            success = chatbot.initialize()
            
            if success:
                db.save_chatbot(chatbot)
                results.append({'url': url, 'status': 'success'})
            else:
                results.append({'url': url, 'status': 'failed'})
        except Exception as e:
            results.append({'url': url, 'status': f'error: {str(e)}'})
    
    return results
        """, language="python")
    
    with improvement_tabs[3]:
        st.subheader("🎨 Nice-to-Have Features")
        
        st.markdown("### 1. Customizable Chatbot Appearance")
        st.code("""
# Let users customize colors, logo, greeting
class ChatbotConfig:
    def __init__(self):
        self.primary_color = "#0066cc"
        self.secondary_color = "#ffffff"
        self.logo_url = None
        self.greeting_message = "Hi! How can I help?"
        self.placeholder_text = "Ask me anything..."

def render_custom_chatbot(chatbot, config):
    st.markdown(f'''
    <style>
    .chatbot-container {{
        background-color: {config.primary_color};
        color: {config.secondary_color};
    }}
    </style>
    ''', unsafe_allow_html=True)
    
    if config.logo_url:
        st.image(config.logo_url, width=100)
    
    st.markdown(f"### {config.greeting_message}")
        """, language="python")
        
        st.markdown("### 2. Multi-language Support")
        st.code("""
# Detect language and respond accordingly
from langdetect import detect

def ask_multilingual(self, question: str) -> str:
    # Detect question language
    lang = detect(question)
    
    # Add language instruction to prompt
    prompt = f'''You are a helpful AI assistant for {self.company_name}.

Respond in {lang} language.

CONTEXT: {context}
QUESTION: {question}

Answer in {lang}:'''
    
    return self.ai.call_llm(prompt)
        """, language="python")
        
        st.markdown("### 3. Voice Input/Output")
        st.code("""
# Add speech-to-text and text-to-speech
import speech_recognition as sr
from gtts import gTTS
import io

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            return None

def voice_output(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    st.audio(fp)
        """, language="python")
        
        st.markdown("### 4. Export Chat History")
        st.code("""
# Let users download their conversation
import json
from datetime import datetime

def export_chat_history():
    history = st.session_state.chat_history
    
    export_data = {
        'chatbot': st.session_state.current_company,
        'exported_at': datetime.now().isoformat(),
        'messages': history
    }
    
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="📥 Download Chat History",
        data=json_str,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
        """, language="python")

st.markdown("---")
st.info("💡 **Tip**: Start with Priority 1 improvements to make the core functionality work, then gradually add Priority 2 and 3 features based on user needs.")
