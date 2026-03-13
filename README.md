# 🎓 ScholarBot — AI Research & Education Assistant

**NeoStats AI Engineer Case Study**

An intelligent chatbot that helps students, researchers, and educators understand complex academic content, explore research papers, and get answers from uploaded study materials — powered by RAG and live web search.

---

## ✨ Features

| Feature | Details |
|---|---|
| **📄 RAG Integration** | Upload PDF, DOCX, TXT → auto-chunked → embedded with `all-MiniLM-L6-v2` → FAISS vector search |
| **🌐 Live Web Search** | Serper.dev (Google Search) + Tavily fallback — real-time answers |
| **⚡ Response Modes** | **Concise** (2–4 sentences) or **Detailed** (structured, with headings) |
| **🤖 Multi-LLM** | Groq (Llama3), OpenAI (GPT-4o-mini), Google Gemini — switchable live |
| **📎 Source Citations** | Every answer shows exactly which document or URL was used |
| **🎨 Dark UI** | Custom-styled Streamlit with professional dark theme |

---

## 📁 Project Structure

```
project/
├── config/
│   └── config.py          ← All API keys, settings, system prompt
├── models/
│   ├── llm.py             ← LLM abstraction (OpenAI / Groq / Gemini)
│   └── embeddings.py      ← HuggingFace sentence-transformers wrapper
├── utils/
│   ├── rag.py             ← Document ingestion, chunking, FAISS store, retrieval
│   ├── web_search.py      ← Serper + Tavily search integration
│   └── chat.py            ← Prompt building, history trimming
├── app.py                 ← Main Streamlit UI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Local Setup

```bash
# 1. Clone and enter project
git clone https://github.com/your-username/scholarbot-neostats
cd scholarbot-neostats/project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env with your keys

# 4. Run
streamlit run app.py
```

---

## 🔑 API Keys

| Key | Where to get | Notes |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free, fast |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Optional |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Free |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) | 2500 free queries |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | Web search fallback |

**Minimum to run:** Set at least one LLM key (Groq recommended — it's free).

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Point to `project/app.py`
4. Add your API keys under **Settings → Secrets**:

```toml
GROQ_API_KEY = "your_key"
SERPER_API_KEY = "your_key"
```

---

## 📝 Use Case

ScholarBot solves a real problem: academic content is dense, time-consuming to parse, and hard to search. By combining RAG (for uploaded materials) and live web search (for current information), ScholarBot lets anyone:

- Get instant summaries of uploaded research papers
- Ask specific questions about textbook chapters
- Explore topics with cited, up-to-date web results
- Switch between quick answers (Concise) and deep dives (Detailed)
