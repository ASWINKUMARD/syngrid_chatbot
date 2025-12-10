###########################################################
#                AI CHATBOT LEAD GENERATOR                #
#              FULL FINAL FIXED WORKING VERSION           #
###########################################################

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re, hashlib, time, json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

############################################################
# CONFIG
############################################################

class Config:
    OPENROUTER_API_KEY = ""  #  <-- ADD YOUR KEY HERE (Required)
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    MODELS = {
        "free": "kwaipilot/kat-coder-pro:free",
        "cheap": "anthropic/claude-3-haiku",
        "smart": "anthropic/claude-3-sonnet"
    }
    CURRENT_MODEL = "free"
    DEPLOY_URL = "http://localhost:8501"
    SCRAPE_TIMEOUT = 8
    MAX_CONTENT_LENGTH = 5000
    QUESTIONS_BEFORE_CAPTURE = 3

    @classmethod
    def model(cls):return cls.MODELS[cls.CURRENT_MODEL]


############################################################
# STORAGE (Session Based)
############################################################

def init_state():
    ss=st.session_state
    ss.setdefault("storage_chatbots",{})
    ss.setdefault("storage_leads",[])
    ss.setdefault("active_bots",{})
    ss.setdefault("current_bot",None)
    ss.setdefault("chat",[])
    ss.setdefault("count",0)
    ss.setdefault("lead_step",None)
    ss.setdefault("lead_data",{})
    ss.setdefault("session",hashlib.md5(str(time.time()).encode()).hexdigest()[:10])


############################################################
# SCRAPER
############################################################

class Scraper:
    def scrape_page(self,url):
        try:
            r=requests.get(url,headers={"User-Agent":"Mozilla"},timeout=8)
            if r.status_code!=200:return None
            soup=BeautifulSoup(r.text,'html.parser')
            for x in soup(['script','style','footer','header','nav']):x.decompose()
            text=soup.get_text("\n",strip=True)
            lines=[l for l in text.split("\n") if 30<len(l)<400]
            return "\n".join(lines)[:Config.MAX_CONTENT_LENGTH]
        except:return None

    def scrape(self,url,progress=None):
        if not url.startswith("http"):url="https://"+url
        urls=[url,f"{url}/about",f"{url}/services",f"{url}/products",f"{url}/contact"]
        pages=[];all=""
        with ThreadPoolExecutor(3) as ex:
            tasks={ex.submit(self.scrape_page,u):u for u in urls}
            for i,f in enumerate(as_completed(tasks),1):
                if progress:progress(i,len(tasks),tasks[f])
                res=f.result()
                if res:pages.append(res);all+=res
        
        emails=re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}",all)[:3]
        phones=re.findall(r"\+?\d[\d\s\-()]{9,}",all)[:3]
        return pages,{"emails":emails,"phones":phones}


############################################################
# AI ENGINE
############################################################

class AI:
    def ask(self,prompt):
        if not Config.OPENROUTER_API_KEY: return "❗ Add OpenRouter API key first."
        try:
            r=requests.post(Config.OPENROUTER_API_URL,
                headers={"Authorization":f"Bearer {Config.OPENROUTER_API_KEY}",
                         "Content-Type":"application/json"},
                json={"model":Config.model(),"messages":[{"role":"user","content":prompt}]})
            return r.json()["choices"][0]["message"]["content"]
        except:return "⚠ AI request failed."


############################################################
# BOT
############################################################

class Bot:
    def __init__(self,id,name,url):
        self.id=id;self.name=name;self.url=url
        self.scraper=Scraper();self.ai=AI()
        self.pages=[];self.info={};self.ready=False

    def init(self,progress=None):
        self.pages,self.info=self.scraper.scrape(self.url,progress)
        self.ready=bool(self.pages);return self.ready

    def chat(self,q):
        if not self.ready:return "Bot loading..."
        if any(x in q.lower() for x in["email","contact","phone"]):
            return f"📧 {self.info.get('emails',[])}\n📞 {self.info.get('phones',[])}\n🌐 {self.url}"

        ctx="\n".join(p[:800] for p in self.pages[:3])
        return self.ai.ask(f"Context:{ctx}\nUser:{q}\nReply business friendly.")


############################################################
# SUPPORT
############################################################

def valid_url(u):return bool(re.match(r"^(https?://)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}",u))
def bot_id(name,url):return hashlib.md5(f"{name}{url}{time.time()}".encode()).hexdigest()[:12]


############################################################
# EMBED MODE (Widget)
############################################################

def embed_ui():
    id=st.query_params.get("id")
    if not id:return st.error("Invalid chatbot URL")
    data=st.session_state.storage_chatbots.get(id)
    if not data:return st.error("Bot not found")

    if id not in st.session_state.active_bots:
        bot=Bot(id,data['name'],data['url'])
        with st.spinner("Loading..."):
            if bot.init():st.session_state.active_bots[id]=bot
            else:return st.error("Scrape failed")
    bot=st.session_state.active_bots[id]

    st.write(f"### 💬 {bot.name}")
    for m in st.session_state.chat:st.chat_message(m["role"]).write(m["msg"])

    if q:=st.chat_input("Message..."):
        st.session_state.chat.append({"role":"user","msg":q})
        with st.spinner("Thinking..."):a=bot.chat(q)
        st.session_state.chat.append({"role":"assistant","msg":a})
        st.rerun()


############################################################
# MAIN DASHBOARD
############################################################

def main():
    if st.query_params.get("mode")=="embed":return embed_ui()

    st.set_page_config(page_title="AI Chatbot Lead Generator",page_icon="🤖",layout="wide")
    init_state()
    st.sidebar.title("Chatbot Manager")

    # CREATE BOT
    with st.sidebar.expander("➕ Create New Bot",True):
        name=st.text_input("Business Name")
        url=st.text_input("Website URL")
        if st.button("Create"):
            if not(name and valid_url(url)):st.error("Enter valid details")
            else:
                id=bot_id(name,url)
                b=Bot(id,name,url)
                prog=st.progress(0);txt=st.empty()
                def cb(c,t,u):prog.progress(c/t);txt.write(f"Scraping {u}")

                if b.init(cb):
                    st.session_state.active_bots[id]=b
                    st.session_state.storage_chatbots[id]={"name":name,"url":url}
                    st.session_state.current_bot=id;st.session_state.chat=[]
                    st.success("Bot ready!"); st.rerun()
                else:st.error("Unable to scrape site.")

    # LIST BOTS
    st.sidebar.subheader("Your Bots")
    for id,d in st.session_state.storage_chatbots.items():
        if st.sidebar.button(d["name"]):st.session_state.current_bot=id;st.session_state.chat=[];st.rerun()

    if not st.session_state.current_bot: return st.info("➡ Create or select a bot")

    bot_obj=st.session_state.active_bots.get(st.session_state.current_bot)
    if not bot_obj: return st.error("Bot data missing")

    st.write(f"## Chat with **{bot_obj.name}**")

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["msg"])

    if q:=st.chat_input("Ask anything..."):
        st.session_state.chat.append({"role":"user","msg":q})
        with st.spinner("Thinking..."):a=bot_obj.chat(q)
        st.session_state.chat.append({"role":"assistant","msg":a})
        st.rerun()


############################################################
if __name__=="__main__":
    main()
############################################################
