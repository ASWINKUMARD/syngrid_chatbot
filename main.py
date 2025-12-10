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

# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    OPENROUTER_API_KEY = ""  # <---- ADD YOUR KEY HERE
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

    MODELS = {
        "free": "kwaipilot/kat-coder-pro:free",
        "cheap": "anthropic/claude-3-haiku",
        "smart": "anthropic/claude-3-sonnet"
    }
    CURRENT_MODEL = "free"

    DEPLOYMENT_URL = "http://localhost:8501"

    SCRAPE_TIMEOUT = 8
    MAX_PAGES = 15
    MAX_CONTENT_LENGTH = 5000

    QUESTIONS_BEFORE_CAPTURE = 3

    APP_TITLE = "🤖 AI Chatbot Lead Generator Pro"
    APP_ICON = "🤖"

    @classmethod
    def get_model(cls):
        return cls.MODELS.get(cls.CURRENT_MODEL)

    @classmethod
    def validate(cls):
        issues = []
        if not cls.OPENROUTER_API_KEY:
            issues.append("❌ API Key not set")
        elif not cls.OPENROUTER_API_KEY.startswith("sk-or-v1-"):
            issues.append("⚠️ API key format looks incorrect")
        return issues


# =====================================================================
# STORAGE
# =====================================================================

class PersistentStorage:
    def __init__(self):
        self._init_storage()

    def _init_storage(self):
        ss = st.session_state
        ss.setdefault('storage_leads', [])
        ss.setdefault('storage_chatbots', {})
        ss.setdefault('storage_next_lead_id', 1)

    def save_lead(self, chatbot_id, company_name, user_name, user_email,
                  user_phone, session_id, questions_asked, conversation):
        try:
            lead = {
                'id': st.session_state.storage_next_lead_id,
                'chatbot_id': chatbot_id,
                'company_name': company_name,
                'user_name': user_name,
                'user_email': user_email,
                'user_phone': user_phone,
                'session_id': session_id,
                'questions_asked': questions_asked,
                'conversation': json.dumps(conversation),
                'created_at': datetime.now().isoformat(),
                'status': 'new'
            }
            st.session_state.storage_leads.append(lead)
            st.session_state.storage_next_lead_id += 1
            return True
        except:
            return False

    def get_leads(self, bot=None):
        return [l for l in st.session_state.storage_leads if (not bot or l['chatbot_id']==bot)]

    def save_chatbot(self, id,name,url,embed):
        st.session_state.storage_chatbots[id]={
            'chatbot_id':id,'company_name':name,'website_url':url,
            'embed_code':embed,'created_at':datetime.now().isoformat(),
            'status':'active','total_leads':0,'total_questions':0
        }

    def export_leads_csv(self,bot=None):
        leads=self.get_leads(bot)
        if not leads:return ""
        out="ID,Company,Name,Email,Phone,Questions,Date,Status\n"
        for l in leads:
            out+=f"{l['id']},{l['company_name']},{l['user_name']},{l['user_email']},{l['user_phone']},{l['questions_asked']},{l['created_at']},{l['status']}\n"
        return out

    def update_chatbot_stats(self, bot, questions_inc=0, leads_inc=0):
        if bot in st.session_state.storage_chatbots:
            d=st.session_state.storage_chatbots[bot]
            d['total_questions']+=questions_inc
            d['total_leads']+=leads_inc


# =====================================================================
# SCRAPER
# =====================================================================

class SmartScraper:
    def __init__(self):
        self.headers={"User-Agent":"Mozilla/5.0"}
        self.timeout=Config.SCRAPE_TIMEOUT

    def scrape_page(self,url):
        try:
            r=requests.get(url,headers=self.headers,timeout=self.timeout)
            if r.status_code!=200:return None
            soup=BeautifulSoup(r.text,'html.parser')
            for t in soup(['script','style','nav','footer','header']): t.decompose()
            lines=[l.strip() for l in soup.get_text("\n",strip=True).split("\n")
                   if 20<len(l.strip())<500]
            text="\n".join(lines)[:Config.MAX_CONTENT_LENGTH]
            return {"url":url,"content":text,"length":len(text)}
        except:return None

    def scrape_website(self,url,progress=None):
        if not url.startswith("http"):url="https://"+url
        urls=[url,f"{url}/about",f"{url}/services",f"{url}/products",f"{url}/contact"]
        pages=[];all_text=""
        with ThreadPoolExecutor(3) as ex:
            tasks={ex.submit(self.scrape_page,u):u for u in urls[:Config.MAX_PAGES]}
            for i,f in enumerate(as_completed(tasks),1):
                if progress:progress(i,len(tasks),tasks[f])
                res=f.result()
                if res:pages.append(res);all_text+="\n"+res['content']

        emails=re.findall(r"[a-zA-Z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",all_text)[:3]
        phones=re.findall(r"\+?\d[\d\s\-()]{9,}",all_text)[:3]

        return pages,{"emails":emails,"phones":phones}


# =====================================================================
# AI Engine
# =====================================================================

class AIEngine:
    def call_api(self,prompt):
        if not Config.OPENROUTER_API_KEY:
            return "❌ API Key missing, add it inside Config.OPENROUTER_API_KEY"

        try:
            r=requests.post(Config.OPENROUTER_API_BASE,
                headers={
                    "Authorization":f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type":"application/json"
                },
                json={"model":Config.get_model(),
                      "messages":[{"role":"user","content":prompt}],
                      "max_tokens":600,"temperature":0.7})
            return r.json()["choices"][0]["message"]["content"]
        except:return "⚠ API error, try again."


# =====================================================================
# CHATBOT
# =====================================================================

class SmartChatbot:
    def __init__(self,id,name,url):
        self.id=id;self.company=name;self.url=url
        self.scraper=SmartScraper();self.ai=AIEngine()
        self.pages=[];self.contacts={};self.ready=False

    def initialize(self,progress=None):
        self.pages,self.contacts=self.scraper.scrape_website(self.url,progress)
        self.ready=bool(self.pages);return self.ready

    def ask(self,q):
        if not self.ready:return "Initializing chatbot..."
        if any(x in q.lower() for x in ["contact","email","phone"]):
            out=f"📧 Emails: {', '.join(self.contacts.get('emails',[]))}\n"
            out+=f"📱 Phones: {', '.join(self.contacts.get('phones',[]))}\n"
            return out+f"🌐 Website: {self.url}"

        context="\n".join(p['content'][:1000] for p in self.pages[:3])
        prompt=f"You are an AI assistant for {self.company}.\n{context}\nUser:{q}\nAnswer briefly."
        return self.ai.call_api(prompt)


# =====================================================================
# Utilities
# =====================================================================

def validate_email(e):return bool(re.match(r"[^@]+@[^@]+\.[^@]+",e))
def validate_url(u):return bool(re.match(r"^(https?://)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}",u))
def generate_id(name,url):return hashlib.md5(f"{name}{url}{time.time()}".encode()).hexdigest()[:16]

def init_state():
    defaults={
        "storage":PersistentStorage(),
        "active":{}, "current":None,
        "chat_history":[], "count":0,
        "lead_step":None,"lead_data":{},
        "session":hashlib.md5(str(time.time()).encode()).hexdigest()[:10],
        "lead_done":False,
    }
    for k,v in defaults.items():st.session_state.setdefault(k,v)


# =====================================================================
# EMBED MODE UI
# =====================================================================

def render_embed():
    st.set_page_config(page_title="Chat",page_icon="💬",layout="wide")

    bot_id=st.query_params.get("id")
    if not bot_id:return st.error("Invalid chatbot")

    storage=st.session_state.storage
    data=storage.storage_chatbots.get(bot_id)
    if not data:return st.error("Chatbot not found")

    if bot_id not in st.session_state.active:
        bot=SmartChatbot(bot_id,data['company_name'],data['website_url'])
        with st.spinner("Loading..."):
            if bot.initialize():st.session_state.active[bot_id]=bot
            else:return st.error("Failed to load bot")

    bot=st.session_state.active[bot_id]
    st.markdown(f"### 💬 {bot.company}")

    for m in st.session_state.chat_history:
        with st.chat_message(m['role']):st.markdown(m['content'])

    if q:=st.chat_input("Type message..."):
        st.session_state.chat_history+=[{"role":"user","content":q}]
        with st.spinner("Thinking..."):a=bot.ask(q)
        st.session_state.chat_history+=[{"role":"assistant","content":a}]
        st.rerun()


# =====================================================================
# MAIN APP UI
# =====================================================================

def main():
    if st.query_params.get("mode")=="embed":return render_embed()

    st.set_page_config(page_title=Config.APP_TITLE,page_icon=Config.APP_ICON,layout="wide")
    init_state()

    st.markdown("""
    <style>.stApp {max-width:1400px;margin:auto;}</style>
    """,unsafe_allow_html=True)

    storage=st.session_state.storage

    # ---------------- Sidebar ----------------
    st.sidebar.title("Control Panel")
    issues=Config.validate()
    if issues:[st.sidebar.warning(i) for i in issues]
    else:st.sidebar.success("Config OK")

    with st.sidebar.expander("➕ Create Chatbot",True):
        n=st.text_input("Company")
        u=st.text_input("Website")
        if st.button("Create"):
            if not(n and validate_url(u)):st.error("Valid fields required")
            else:
                id=generate_id(n,u)
                bot=SmartChatbot(id,n,u)
                prog=st.progress(0);txt=st.empty()
                def cb(c,t,url):prog.progress(c/t);txt.write(f"Scraping {url}")
                if bot.initialize(cb):
                    st.session_state.active[id]=bot
                    embed=f"<iframe src='{Config.DEPLOYMENT_URL}?mode=embed&id={id}' width='400' height='600'></iframe>"
                    storage.save_chatbot(id,n,u,embed)
                    st.session_state.current=id;st.rerun()
                else:st.error("Scrape failed")

    st.sidebar.markdown("---")
    for id,b in storage.storage_chatbots.items():
        if st.sidebar.button("💬 "+b['company_name']):st.session_state.current=id;st.session_state.chat_history=[];st.rerun()

    # ---------------- Main Screen ----------------
    if not st.session_state.current:
        st.write("## Create/Select a chatbot to start.")
        return

    bot=storage.storage_chatbots[st.session_state.current]
    if st.session_state.current not in st.session_state.active:
        b=SmartChatbot(bot['chatbot_id'],bot['company_name'],bot['website_url'])
        with st.spinner("Loading..."):
            if b.initialize():st.session_state.active[bot['chatbot_id']]=b
            else:return st.error("Init failed")

    active=st.session_state.active[st.session_state.current]
    st.write(f"### 💬 Chatting with **{active.company}**")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):st.write(msg['content'])

    if q:=st.chat_input("Ask..."):
        st.session_state.chat_history.append({"role":"user","content":q})
        with st.spinner("Thinking..."):
            a=active.ask(q)
        st.session_state.chat_history.append({"role":"assistant","content":a})
        st.rerun()


if __name__=="__main__":
    main()
