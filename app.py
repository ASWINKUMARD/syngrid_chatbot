import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator
import sqlite3
from contextlib import contextmanager, asynccontextmanager
import gc

# Memory optimization: Disable some torch features
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise Exception("OPENROUTER_API_KEY is missing! Add it in Render Environment Variables.")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODELS = ["kwaipilot/kat-coder-pro:free"]

# Global variables
retriever = None
cache = {}
status = {"ready": False, "message": "Initializing..."}
DB_NAME = "users.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print("✓ Database initialized")

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

class UserInfoRequest(BaseModel):
    name: str
    email: str
    phone: str
    session_id: str = "default"

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v):
        if not re.match(r"^\+?[0-9\s\-\(\)]{10,}$", v):
            raise ValueError("Invalid phone number")
        return v

# Optimized scraping - reduced pages for memory
def scrape_website(base_url, max_pages=15):
    visited = set()
    all_content = []

    def crawl(url, depth=0):
        if len(visited) >= max_pages or depth > 2:
            return
        if url in visited:
            return

        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if resp.status_code != 200:
                return

            visited.add(url)
            print(f"✓ Scraped [{len(visited)}/{max_pages}]: {url}")

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            cleaned = "\n".join([x.strip() for x in text.split("\n") if x.strip()])

            if len(cleaned) > 100:
                all_content.append(cleaned)

            # Clear soup to free memory
            soup.decompose()
            del soup
            
            base_domain = urlparse(base_url).netloc
            for link in BeautifulSoup(resp.text, "html.parser").find_all("a", href=True)[:20]:
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_domain:
                    crawl(next_url, depth + 1)
                    
        except Exception as e:
            print(f"✗ Error: {e}")

    crawl(base_url)
    gc.collect()
    return "\n\n".join(all_content)

def initialize_rag(url="https://syngrid.com/"):
    global retriever, status
    try:
        print("\n" + "="*50)
        print("INITIALIZING RAG CHATBOT")
        print("="*50)
        
        status["message"] = "Scraping website..."
        content = scrape_website(url, max_pages=15)

        if len(content) < 500:
            status["message"] = "Failed: Not enough content scraped"
            return

        status["message"] = "Splitting content..."
        # Optimized chunk size for memory
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )
        chunks = splitter.split_text(content)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Clear content to free memory
        del content
        gc.collect()

        status["message"] = "Loading embeddings..."
        # Use smaller, faster embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        status["message"] = "Building vector database..."
        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Clear chunks to free memory
        del chunks
        gc.collect()

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Reduced from 4 to 3 for speed
        )

        status["ready"] = True
        status["message"] = "Ready!"
        print("✓ RAG Chatbot Ready!")
        print("="*50 + "\n")
        
        # Final memory cleanup
        gc.collect()

    except Exception as e:
        status["message"] = f"Error: {str(e)}"
        print(f"✗ Initialization Error: {e}")

def call_llm(question, context):
    prompt = f"""Using only the following context, answer the question in 1-3 sentences.

Context:
{context[:2500]}

Question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODELS[0],
        "messages": [
            {"role": "system", "content": "You answer ONLY using given context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }

    try:
        res = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"LLM Error: {res.status_code}")
            return "Unable to generate response."
    except Exception as e:
        print(f"LLM Exception: {e}")
        return "Error contacting LLM."

# Session tracking
session_data = {}

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    initialize_rag("https://syngrid.com/")
    yield
    # Shutdown (cleanup)
    gc.collect()

app = FastAPI(title="Syngrid AI Assistant", lifespan=lifespan)

@app.get("/api/status")
def api_status():
    return JSONResponse(content=status)

@app.post("/api/ask")
def ask(req: QuestionRequest):
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status["message"])

    sid = req.session_id
    if sid not in session_data:
        session_data[sid] = {"count": 0, "info": False}

    if session_data[sid]["count"] >= 3 and not session_data[sid]["info"]:
        return JSONResponse(content={
            "answer": "⚠️ Please provide your information to continue asking questions.",
            "needs_user_info": True
        })

    q = req.question.lower().strip()

    if q in cache:
        answer = f"💾 {cache[q]}"
    else:
        docs = retriever.invoke(req.question)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            answer = call_llm(req.question, context)
            cache[q] = answer
        else:
            answer = "❌ No relevant information found."

    session_data[sid]["count"] += 1
    
    return JSONResponse(content={
        "answer": answer,
        "needs_user_info": False,
        "question_count": session_data[sid]["count"]
    })

@app.post("/api/submit-info")
def submit_info(req: UserInfoRequest):
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone) VALUES (?, ?, ?)",
                (req.name, req.email, req.phone)
            )
            conn.commit()
            user_id = cursor.lastrowid

        session_data[req.session_id]["info"] = True
        
        print("\n" + "="*50)
        print("USER INFORMATION SAVED:")
        print(f"ID: {user_id}")
        print(f"Name: {req.name}")
        print(f"Email: {req.email}")
        print(f"Phone: {req.phone}")
        print("="*50 + "\n")
        
        return JSONResponse(content={
            "success": True,
            "message": f"✅ Thank you, {req.name}! Your information has been saved."
        })

    except Exception as e:
        print(f"Database Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save information")

@app.get("/api/users")
def list_users():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
            users = [dict(row) for row in cursor.fetchall()]
        return JSONResponse(content={"users": users, "count": len(users)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def home_page():
    return HTMLResponse(content=get_html_content())

def get_html_content():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Syngrid AI Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            width: 100%;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .status {
            padding: 15px 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #28a745;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            padding: 20px 30px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.4;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.bot .message-content {
            background: white;
            border: 1px solid #e9ecef;
            color: #333;
        }
        .input-area {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e9ecef;
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        input, button {
            padding: 12px 20px;
            border: 1px solid #e9ecef;
            border-radius: 25px;
            font-size: 14px;
        }
        input {
            flex: 1;
            outline: none;
        }
        input:focus { border-color: #667eea; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); }
        button:active { transform: scale(0.95); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .user-info-form {
            display: none;
            padding: 30px;
            background: white;
            border-radius: 15px;
            margin: 20px 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .user-info-form.show { display: block; }
        .user-info-form h3 { margin-bottom: 20px; color: #667eea; }
        .form-group { margin-bottom: 15px; }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        .form-group input { width: 100%; border-radius: 10px; }
        .examples {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e9ecef;
        }
        .examples h4 { margin-bottom: 10px; color: #667eea; }
        .examples ul { list-style: none; padding-left: 0; }
        .examples li {
            padding: 8px 0;
            color: #666;
            cursor: pointer;
            transition: color 0.2s;
        }
        .examples li:hover { color: #667eea; }
        .error-message { color: #dc3545; font-size: 13px; margin-top: 5px; }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Syngrid AI Assistant</h1>
            <p>RAG-powered chatbot trained on Syngrid website content</p>
        </div>
        
        <div class="status">
            <div class="status-indicator"></div>
            <span id="status-text">Initializing...</span>
        </div>
        
        <div class="chat-container" id="chat-container">
            <div class="message bot">
                <div class="message-content">
                    👋 Hello! I'm the Syngrid AI Assistant. Ask me anything about Syngrid!
                </div>
            </div>
        </div>
        
        <div class="user-info-form" id="user-info-form">
            <h3>📋 Please provide your information to continue</h3>
            <div class="form-group">
                <label>Name</label>
                <input type="text" id="user-name" placeholder="Enter your full name" required>
                <div class="error-message" id="name-error"></div>
            </div>
            <div class="form-group">
                <label>Email</label>
                <input type="email" id="user-email" placeholder="Enter your email address" required>
                <div class="error-message" id="email-error"></div>
            </div>
            <div class="form-group">
                <label>Phone Number</label>
                <input type="tel" id="user-phone" placeholder="Enter your phone number" required>
                <div class="error-message" id="phone-error"></div>
            </div>
            <button onclick="submitUserInfo()" id="submit-info-btn">Submit Information</button>
        </div>
        
        <div class="input-area">
            <div class="input-group">
                <input type="text" id="user-input" placeholder="Type your question here..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" id="send-btn">Send</button>
            </div>
        </div>
        
        <div class="examples">
            <h4>Example Questions:</h4>
            <ul>
                <li onclick="askExample('What services does Syngrid offer?')">• What services does Syngrid offer?</li>
                <li onclick="askExample('Tell me about Syngrid technology')">• Tell me about Syngrid technology</li>
                <li onclick="askExample('How can I contact Syngrid?')">• How can I contact Syngrid?</li>
            </ul>
        </div>
    </div>

    <script>
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('status-text').textContent = 
                    data.ready ? '✅ Ready' : '⏳ ' + data.message;
            } catch (error) {
                console.error('Status check error:', error);
            }
        }
        
        checkStatus();
        setInterval(checkStatus, 5000);
        
        function addMessage(content, isUser) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            const escaped = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            messageDiv.innerHTML = '<div class="message-content">' + escaped + '</div>';
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const question = input.value.trim();
            const sendBtn = document.getElementById('send-btn');
            
            if (!question) return;
            
            addMessage(question, true);
            input.value = '';
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<div class="loading"></div>';
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, session_id: sessionId })
                });
                
                const data = await response.json();
                
                if (data.needs_user_info) {
                    document.getElementById('user-info-form').classList.add('show');
                }
                
                addMessage(data.answer, false);
                
            } catch (error) {
                addMessage('❌ Error: ' + error.message, false);
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        }
        
        async function submitUserInfo() {
            const name = document.getElementById('user-name').value.trim();
            const email = document.getElementById('user-email').value.trim();
            const phone = document.getElementById('user-phone').value.trim();
            const submitBtn = document.getElementById('submit-info-btn');
            
            document.getElementById('name-error').textContent = '';
            document.getElementById('email-error').textContent = '';
            document.getElementById('phone-error').textContent = '';
            
            if (name.length < 2) {
                document.getElementById('name-error').textContent = 'Name must be at least 2 characters';
                return;
            }
            
            if (!email.match(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/)) {
                document.getElementById('email-error').textContent = 'Please enter a valid email';
                return;
            }
            
            if (phone.length < 10) {
                document.getElementById('phone-error').textContent = 'Please enter a valid phone number';
                return;
            }
            
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
            
            try {
                const response = await fetch('/api/submit-info', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, email, phone, session_id: sessionId })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('user-info-form').classList.remove('show');
                    addMessage(data.message, false);
                    document.getElementById('user-name').value = '';
                    document.getElementById('user-email').value = '';
                    document.getElementById('user-phone').value = '';
                }
                
            } catch (error) {
                alert('Error submitting information: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit Information';
            }
        }
        
        function askExample(question) {
            document.getElementById('user-input').value = question;
            sendMessage();
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
