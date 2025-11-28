import os
import json
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
import sqlite3
from contextlib import contextmanager
from typing import Optional

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "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"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

MODELS = ["kwaipilot/kat-coder-pro:free"]

# Initialize FastAPI
app = FastAPI(title="Syngrid AI Assistant")

# Global variables
retriever = None
cache = {}
status = {"ready": False, "message": "Initializing..."}

# Database setup
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
    """Initialize SQLite database for storing user information"""
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

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

class UserInfoRequest(BaseModel):
    name: str
    email: str
    phone: str
    session_id: str = "default"
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        if not re.match(r'^\+?[0-9\s\-\(\)]{10,}$', v):
            raise ValueError('Invalid phone number format')
        return v

# Scraping and RAG functions
def scrape_website(base_url, max_pages=25):
    visited = set()
    all_content = []

    def crawl(url, depth=0):
        if len(visited) >= max_pages or depth > 3:
            return
        if url in visited:
            return

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return

            visited.add(url)
            print(f"✓ Scraped [{len(visited)}/{max_pages}]: {url}")

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            clean_text = "\n".join(lines)

            if len(clean_text) > 100:
                all_content.append(clean_text)

            base_domain = urlparse(base_url).netloc
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_domain:
                    crawl(next_url, depth + 1)

        except Exception as e:
            print(f"✗ Error scraping {url}: {e}")

    crawl(base_url)
    return "\n\n".join(all_content)

def initialize_rag(url="https://syngrid.com/"):
    global retriever, status

    try:
        print("\n" + "="*50)
        print("INITIALIZING RAG CHATBOT")
        print("="*50)

        status["message"] = "Scraping website..."
        content = scrape_website(url)

        if len(content) < 500:
            status["message"] = "Error: Not enough content scraped"
            return

        status["message"] = "Splitting content..."
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=150
        )
        chunks = splitter.split_text(content)
        print(f"✓ Created {len(chunks)} chunks")

        status["message"] = "Loading embeddings..."
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        status["message"] = "Building vector database..."
        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        status["ready"] = True
        status["message"] = "Ready!"
        print("✓ RAG Chatbot Ready!\n")

    except Exception as e:
        print(f"Initialization Error: {e}")
        status["message"] = f"Error: {str(e)}"

def call_llm(question, context):
    prompt = f"""Using only the following context, answer the question in 1-3 sentences.

Context:
{context[:3000]}

Question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    for model in MODELS:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You answer ONLY using given context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 120
            }

            res = requests.post(
                OPENROUTER_API_BASE,
                headers=headers,
                json=payload,
                timeout=30
            )

            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"{model} failed: {e}")

    return "Error: All models failed."

# Session tracking
session_data = {}

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database and RAG on startup"""
    init_database()
    initialize_rag("https://syngrid.com/")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    return HTMLResponse(content=get_html_content())

@app.get("/api/status")
async def get_status_endpoint():
    """Get chatbot initialization status"""
    return JSONResponse(content=status)

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Handle question requests"""
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status["message"])

    session_id = request.session_id
    
    # Initialize session if not exists
    if session_id not in session_data:
        session_data[session_id] = {"question_count": 0, "info_collected": False}
    
    # Check if user info needed
    if session_data[session_id]["question_count"] >= 3 and not session_data[session_id]["info_collected"]:
        return JSONResponse(content={
            "answer": "⚠️ Please provide your information to continue asking questions.",
            "needs_user_info": True
        })
    
    # Process question
    q_lower = request.question.lower().strip()
    
    if q_lower in cache:
        answer = f"💾 (Cached) {cache[q_lower]}"
    else:
        docs = retriever.invoke(request.question)
        if not docs:
            answer = "❌ No relevant information found."
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = call_llm(request.question, context)
            cache[q_lower] = answer
    
    # Increment question count
    session_data[session_id]["question_count"] += 1
    
    return JSONResponse(content={
        "answer": answer,
        "needs_user_info": False,
        "question_count": session_data[session_id]["question_count"]
    })

@app.post("/api/submit-info")
async def submit_user_info(request: UserInfoRequest):
    """Handle user information submission"""
    
    # Save to database
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone) VALUES (?, ?, ?)",
                (request.name, request.email, request.phone)
            )
            conn.commit()
            user_id = cursor.lastrowid
        
        # Mark session as info collected
        session_id = request.session_id
        if session_id in session_data:
            session_data[session_id]["info_collected"] = True
        
        print("\n" + "="*50)
        print("USER INFORMATION SAVED TO DATABASE:")
        print(f"ID: {user_id}")
        print(f"Name: {request.name}")
        print(f"Email: {request.email}")
        print(f"Phone: {request.phone}")
        print("="*50 + "\n")
        
        return JSONResponse(content={
            "success": True,
            "message": f"✅ Thank you, {request.name}! Your information has been saved."
        })
    
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save user information")

@app.get("/api/users")
async def get_users():
    """Get all stored users (for admin purposes)"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
            users = [dict(row) for row in cursor.fetchall()]
        return JSONResponse(content={"users": users, "count": len(users)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_html_content():
    """Return the HTML content for the chatbot interface"""
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
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .user-info-form {
            display: none;
            padding: 30px;
            background: white;
            border-radius: 15px;
            margin: 20px 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .user-info-form.show { display: block; }
        .user-info-form h3 {
            margin-bottom: 20px;
            color: #667eea;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        .form-group input {
            width: 100%;
            border-radius: 10px;
        }
        .examples {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e9ecef;
        }
        .examples h4 {
            margin-bottom: 10px;
            color: #667eea;
        }
        .examples ul {
            list-style: none;
            padding-left: 0;
        }
        .examples li {
            padding: 8px 0;
            color: #666;
            cursor: pointer;
            transition: color 0.2s;
        }
        .examples li:hover {
            color: #667eea;
        }
        .error-message {
            color: #dc3545;
            font-size: 13px;
            margin-top: 5px;
        }
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
        const statusInterval = setInterval(checkStatus, 5000);
        
        function addMessage(content, isUser) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
            messageDiv.innerHTML = `<div class="message-content">${escapeHtml(content)}</div>`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
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
    import asyncio
    import nest_asyncio
    
    # Allow nested event loops (for Colab/Jupyter)
    nest_asyncio.apply()
    
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting server on port {port}...")
    print(f"📱 Access at: http://localhost:{port}")
    
    # Initialize on startup
    init_database()
    initialize_rag("https://syngrid.com/")
    
    # Run server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        # For Colab/Jupyter environments
        import IPython
        asyncio.create_task(server.serve())
        print(f"\n✅ Server running! Access it at http://localhost:{port}")
        print("⚠️ In Colab, the interface may not be directly accessible.")
        print("💡 For Colab: Deploy to Render for public access, or use ngrok for testing.")
    except:
        # For normal Python environments
        server.run()
