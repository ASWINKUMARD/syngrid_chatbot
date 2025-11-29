"""
RAG Chatbot for Render - Production Ready
Stores ONLY user contact information (Name, Email, Phone)
"""

import os
import time
import requests
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ========================================
# CONFIGURATION
# ========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./contacts.db")

# Fix PostgreSQL URL for Render
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

# ========================================
# DATABASE SETUP (CONTACT INFO ONLY)
# ========================================
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class UserContact(Base):
    """Store ONLY user contact information"""
    __tablename__ = "user_contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True, unique=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Create tables
Base.metadata.create_all(bind=engine)
print("✅ Database initialized - Contact storage ready")

# ========================================
# PYDANTIC MODELS
# ========================================
class QuestionRequest(BaseModel):
    question: str
    session_id: str

class ContactInfoRequest(BaseModel):
    session_id: str
    name: str
    email: EmailStr
    phone: str

class AnswerResponse(BaseModel):
    answer: str
    requires_contact: bool = False
    contact_message: Optional[str] = None
    question_count: int = 0

class ContactResponse(BaseModel):
    message: str
    success: bool

class ContactInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    session_id: str
    name: str
    email: str
    phone: str
    created_at: datetime

class InitRequest(BaseModel):
    url: str = "https://syngrid.com/"
    max_pages: int = 25

class StatusResponse(BaseModel):
    ready: bool
    message: str
    pages_scraped: int = 0

# ========================================
# IN-MEMORY SESSION TRACKING
# ========================================
class SessionManager:
    """Track question counts in memory (no database bloat)"""
    def __init__(self):
        self.sessions = {}  # {session_id: question_count}
    
    def increment(self, session_id: str) -> int:
        """Increment and return question count"""
        self.sessions[session_id] = self.sessions.get(session_id, 0) + 1
        return self.sessions[session_id]
    
    def get_count(self, session_id: str) -> int:
        """Get current question count"""
        return self.sessions.get(session_id, 0)
    
    def reset(self, session_id: str):
        """Reset session after contact submission"""
        self.sessions[session_id] = 0

session_manager = SessionManager()

# ========================================
# RAG SERVICE
# ========================================
class RAGService:
    def __init__(self):
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "message": "Initializing...", "pages_scraped": 0}
        self.embeddings = None

    def scrape_page(self, url, visited, base_domain):
        """Scrape single page"""
        if url in visited:
            return None, []
        
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                return None, []

            visited.add(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            clean_text = "\n".join(lines)

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if urlparse(next_url).netloc == base_domain and next_url not in visited:
                    links.append(next_url)

            return clean_text if len(clean_text) > 100 else None, links

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None, []

    def scrape_website(self, base_url, max_pages=25):
        """Parallel web scraping"""
        visited = set()
        all_content = []
        to_visit = [base_url]
        base_domain = urlparse(base_url).netloc

        print(f"🔍 Scraping {base_url} (max {max_pages} pages)...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=5) as executor:
            while to_visit and len(visited) < max_pages:
                batch = to_visit[:min(5, max_pages - len(visited))]
                to_visit = to_visit[len(batch):]
                
                futures = {
                    executor.submit(self.scrape_page, url, visited, base_domain): url 
                    for url in batch if url not in visited
                }

                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        content, links = future.result()
                        if content:
                            all_content.append(content)
                            print(f"  ✓ [{len(visited)}/{max_pages}] {url[:70]}")
                        
                        for link in links:
                            if link not in visited and link not in to_visit:
                                to_visit.append(link)
                                
                    except Exception as e:
                        print(f"  ✗ Failed: {url[:50]}")

        elapsed = time.time() - start_time
        self.status["pages_scraped"] = len(visited)
        
        print(f"\n✅ Scraped {len(visited)} pages in {elapsed:.1f}s")
        print(f"📊 Content: {len(''.join(all_content)):,} chars\n")

        return "\n\n".join(all_content)

    def initialize(self, url="https://syngrid.com/", max_pages=25):
        """Initialize RAG system"""
        try:
            print("\n" + "="*60)
            print("🚀 INITIALIZING RAG CHATBOT")
            print("="*60)

            # Step 1: Scrape website
            self.status["message"] = "Scraping website..."
            content = self.scrape_website(url, max_pages)

            if len(content) < 500:
                self.status["message"] = "Insufficient content"
                print("❌ Not enough content scraped")
                return False

            # Step 2: Split into chunks
            self.status["message"] = "Processing content..."
            print("\n📄 Splitting into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100
            )
            chunks = splitter.split_text(content)
            print(f"  ✓ Created {len(chunks)} chunks")

            # Step 3: Load embeddings
            self.status["message"] = "Loading AI model..."
            print("\n🧠 Loading embeddings model...")
            
            if self.embeddings is None:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            print("  ✓ Model loaded")

            # Step 4: Build vector database
            self.status["message"] = "Building knowledge base..."
            print("\n💾 Building vector database...")
            vectorstore = Chroma.from_texts(
                chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            print("  ✓ Vector database ready")

            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            self.status["ready"] = True
            self.status["message"] = "Ready"
            print("\n✅ RAG Chatbot initialized successfully!\n")
            print("="*60 + "\n")
            return True

        except Exception as e:
            self.status["message"] = f"Error: {str(e)}"
            print(f"\n❌ Initialization failed: {e}\n")
            return False

    def call_llm(self, question, context):
        """Call OpenRouter API"""
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not configured. Please contact support."

        prompt = f"""Answer the question based ONLY on the context below. Be concise and helpful.

Context:
{context[:3000]}

Question: {question}

Answer (2-4 sentences):"""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Answer questions concisely based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 150
            }

            response = requests.post(
                OPENROUTER_API_BASE,
                headers=headers,
                json=payload,
                timeout=25
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"LLM Error: {response.status_code} - {response.text}")
                return "I apologize, but I'm having trouble generating a response. Please try again."

        except Exception as e:
            print(f"LLM Exception: {e}")
            return "Sorry, I couldn't process your question. Please try again."

    def ask(self, question):
        """Answer question using RAG"""
        if not self.status["ready"]:
            return "⏳ Chatbot is still initializing. Please wait a moment..."

        # Check cache
        q_key = question.lower().strip()
        if q_key in self.cache:
            return self.cache[q_key]

        # Retrieve relevant documents
        try:
            docs = self.retriever.invoke(question)
            if not docs:
                return "I don't have enough information to answer that question."

            context = "\n\n".join([doc.page_content for doc in docs])
            answer = self.call_llm(question, context)

            # Cache result
            self.cache[q_key] = answer
            return answer
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return "I encountered an error processing your question. Please try again."

    def get_status(self):
        return self.status

# Initialize RAG service
rag_service = RAGService()

# ========================================
# FASTAPI APPLICATION
# ========================================
app = FastAPI(
    title="RAG Chatbot API",
    description="AI-powered chatbot with contact collection (Render deployment)",
    version="3.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def has_contact(session_id: str, db: Session) -> bool:
    """Check if contact info exists for session"""
    return db.query(UserContact).filter(
        UserContact.session_id == session_id
    ).first() is not None

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/")
def root():
    """API information"""
    return {
        "name": "RAG Chatbot API",
        "version": "3.0",
        "status": "running",
        "rag_ready": rag_service.status["ready"],
        "description": "Production-ready chatbot with contact collection",
        "endpoints": {
            "status": "GET /status",
            "initialize": "POST /initialize",
            "ask": "POST /ask",
            "submit_contact": "POST /submit-contact",
            "contacts": "GET /contacts",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check for Render"""
    return {
        "status": "healthy",
        "database": "connected",
        "rag_status": rag_service.status["message"]
    }

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Get RAG initialization status"""
    status = rag_service.get_status()
    return StatusResponse(
        ready=status["ready"],
        message=status["message"],
        pages_scraped=status.get("pages_scraped", 0)
    )

@app.post("/initialize", response_model=StatusResponse)
def initialize(request: InitRequest, background_tasks: BackgroundTasks):
    """Initialize RAG with website content"""
    
    if rag_service.status["ready"]:
        return StatusResponse(
            ready=True,
            message="Already initialized",
            pages_scraped=rag_service.status.get("pages_scraped", 0)
        )

    def init_task():
        rag_service.initialize(request.url, request.max_pages)

    background_tasks.add_task(init_task)

    return StatusResponse(
        ready=False,
        message="Initialization started in background",
        pages_scraped=0
    )

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question to the chatbot"""
    
    if not rag_service.status["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Chatbot not ready. Please wait for initialization to complete."
        )

    # Increment question count
    question_count = session_manager.increment(request.session_id)
    
    # Check if contact is required
    has_submitted = has_contact(request.session_id, db)
    requires_contact = (question_count >= 3 and not has_submitted)

    if requires_contact:
        return AnswerResponse(
            answer="",
            requires_contact=True,
            contact_message="To continue chatting, please provide your contact information (name, email, phone number).",
            question_count=question_count
        )

    # Get answer from RAG
    answer = rag_service.ask(request.question)

    return AnswerResponse(
        answer=answer,
        requires_contact=False,
        question_count=question_count
    )

@app.post("/submit-contact", response_model=ContactResponse)
def submit_contact(request: ContactInfoRequest, db: Session = Depends(get_db)):
    """Submit user contact information"""
    
    try:
        # Check if already exists
        existing = db.query(UserContact).filter(
            UserContact.session_id == request.session_id
        ).first()

        if existing:
            return ContactResponse(
                message="Contact information already saved",
                success=True
            )

        # Save new contact
        contact = UserContact(
            session_id=request.session_id,
            name=request.name.strip(),
            email=request.email.strip().lower(),
            phone=request.phone.strip(),
            created_at=datetime.now(timezone.utc)
        )
        
        db.add(contact)
        db.commit()
        db.refresh(contact)

        # Reset question count
        session_manager.reset(request.session_id)

        print(f"✅ Contact saved: {contact.name} ({contact.email})")

        return ContactResponse(
            message="Thank you! Your contact information has been saved.",
            success=True
        )

    except Exception as e:
        db.rollback()
        print(f"❌ Error saving contact: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save contact information. Please try again."
        )

@app.get("/contacts", response_model=List[ContactInfo])
def get_contacts(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all stored contacts (admin endpoint)"""
    
    try:
        contacts = db.query(UserContact)\
            .order_by(UserContact.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
        return contacts
    except Exception as e:
        print(f"Error fetching contacts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve contacts"
        )

@app.get("/contacts/count")
def get_contact_count(db: Session = Depends(get_db)):
    """Get total number of contacts"""
    try:
        count = db.query(UserContact).count()
        return {"total_contacts": count}
    except Exception as e:
        print(f"Error counting contacts: {e}")
        return {"total_contacts": 0}

@app.delete("/contacts/{contact_id}")
def delete_contact(contact_id: int, db: Session = Depends(get_db)):
    """Delete a specific contact"""
    try:
        contact = db.query(UserContact).filter(UserContact.id == contact_id).first()
        if not contact:
            raise HTTPException(status_code=404, detail="Contact not found")
        
        db.delete(contact)
        db.commit()
        return {"message": "Contact deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting contact: {str(e)}")

# ========================================
# STARTUP EVENT - AUTO-INITIALIZE
# ========================================
@app.on_event("startup")
async def startup_event():
    """Auto-initialize chatbot on startup"""
    print("\n" + "="*60)
    print("🚀 RAG CHATBOT STARTING")
    print("="*60)
    print(f"📊 Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'SQLite'}")
    print(f"🔑 API Key: {'✓ Configured' if OPENROUTER_API_KEY else '✗ Missing'}")
    print("="*60 + "\n")
    
    if not rag_service.status["ready"]:
        print("🔄 Auto-initializing with default website...")
        import threading
        thread = threading.Thread(
            target=lambda: rag_service.initialize("https://syngrid.com/", 25)
        )
        thread.start()

# ========================================
# RUN SERVER
# ========================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )