

# 🤖 Basic RAG App — Retrieval-Augmented Generation with Modern AI Stack

> **RAG = Retrieval-Augmented Generation**  
> This app combines document retrieval with LLMs to deliver accurate, context-based answers.


### 🚀 Tech Stack Overview:

| 🧩 Layer          | ⚙️ Technology           | 📝 Description |
|------------------|-------------------------|----------------|
| 💻 Language       | `Python`                | Base language for the app |
| 🔗 Framework      | `FastAPI`               | Lightweight web API framework |
| 🧠 LLM & Orchestration | `LangChain`, `LangGraph` | Manages prompt flows and toolchains |
| 🧮 Embeddings     | `OpenAIEmbeddings`      | Turns text into numerical vectors |
| 📦 Vector Store   | `ChromaDB`              | Used to store and search document chunks |
| 📂 Document Loader | `unstructured`, `markdown` | Handles parsing `.md` files |
| 🧠 Chat Memory    | `LangGraph SQLite Checkpoint` | Maintains session history by user |
| 🤖 Model    | `gpt-4o-mini` | OpenAI ChatGPT model |
| 📜 Logging        | `aiologger`, `logger.py` | Custom async logging system |


## 🗂️ Project Structure

```bash
.
├── main.py                  # 🔥 FastAPI app entry
├── api_tools.py             # 🧠 Core AI logic
├── logger.py                # 🪵 Logging setup
├── docs/                    # 📄 Source markdown files
├── db/
│   ├── chromadb/            # 📦 ChromaDB persistent storage
│   └── chatSession.db       # 💬 SQLite chat memory
├── logs/                    # 📜 Log files
├── .env                     # 🔐 Env variables
└── requirements.txt         # 📦 All required dependencies
```


## 🔧 Environment Configuration (`.env`)

```ini
OPENAI_API_KEY=your_openai_api_key
DB_PATH=db/chromadb/db_name
DB_COLLECTION_NAME=db_collection_name
DOCUMENT_PATH=docs
CHUNK_SIZE=500
CHUNK_OVERLAP=100
CHATSESSION_DATABASE_PATH=db/chatSession.db
```


## 📦 Dependencies (`requirements.txt`)

```txt
fastapi==0.110.0
uvicorn==0.29.0
openai==1.69.0
chromadb==0.6.3
langchain==0.3.20
langchain-core==0.3.45
langchain-text-splitters==0.3.6
langchain_chroma==0.2.2
langchain-community==0.3.19
langgraph==0.3.28
langgraph.checkpoint.sqlite==2.0.6
aiologger==0.7.0
aiofiles==24.1.0
unstructured==0.17.2
markdown==3.7
python-dotenv==1.0.1
pydantic==2.6.4
```

✅ Add this to `requirements.txt` and run:

```bash
pip install -r requirements.txt
```


## 📡 API Reference

### Endpoint: `POST /chat/start`

#### Payload:
```json
{
  "phone_number": "1234567890",
  "question": "Tell me about tourist places in Gujarat."
}
```

#### Response:
```json
{
  "phone_number": "1234567890",
  "message": "Here are some popular places to visit in Gujarat..."
}
```

🧠 Uses vector search to find relevant document snippets and generate a coherent answer using GPT.


## 🧠 AI Flow Overview

```
graph TD
    A[User Input (question + phone number)] --> B[Vector Search with ChromaDB]
    B --> C[Retrieve Top-K Matching Chunks]
    C --> D[Format Prompt Template]
    D --> E[Send to OpenAI API]
    E --> F[Generate AI Response]
    F --> G[Return API Response + Store Chat in SQLite]
```


## 🧠 Core Components

### 1. `logger.py` 📜
Creates a rotating file logger and logs to console and file.

### 2. `api_tools.py` 🧩
Contains the following:
- **VectorStore class**: loads markdown docs, splits them, creates vector embeddings, and stores in ChromaDB.
- **get_vector_store**: lazy loads vector store singleton.
- **Prompt**: a formatted LangChain prompt for OpenAI.
- **create_agent + create_workflow_graph**: builds conversational graph logic using LangGraph.
- **invoke_model**: stores/retrieves session state from SQLite and invokes AI response.
- **chat_start_api**: the actual API logic to handle chat input.

### 3. `main.py` 🚀
Starts FastAPI server and handles incoming chat POST requests.


## 🚀 How It Works

![Image](https://raw.githubusercontent.com/prashant1879/Basic-RAG-App/refs/heads/main/RAG_APPLICATION.png)

### 🧬 Initialization
- `lifespan()` runs on app startup and:
  - Initializes the AI graph with memory
  - Loads the vector DB and documents

### 📤 Chat Flow
1. User hits `/chat/start` with `phone_number` and `question`.
2. Vector search pulls top 2 similar document chunks.
3. A prompt is created with those results.
4. OpenAI generates a response formatted professionally.
5. Result is returned and stored in SQLite memory.


## 💬 Prompt Design

```text
You are a helpful assistant for a tourism chatbot.
Use ONLY the context provided below to answer the user's question.
If not present, refer them to contact info from vector DB.

---
Context:
{context}
---

Question:
{question}

Answer:
Format your response clearly and professionally.
```


## ✅ Supported File Types

📁 All document files in the `docs/` folder must be:
- Markdown format (`.md` only)

> You can extend `VectorStore.legal_file_extensions` to support `.pdf`, `.docx`, etc.


## 🚀 Running the App

1. Create a `.env` file with your values.
2. Add markdown files in the `docs/` folder.
3. Run the app:
   ```bash
   python main.py
   ```

🧪 Test via Postman or any HTTP client:
```http
POST http://127.0.0.1:5000/chat/start
```


## 🔍 Features

✅ Markdown file ingestion  
✅ Vector similarity search  
✅ Prompted GPT-based response  
✅ Phone number-based chat memory  
✅ Logging to both file and console  
✅ Handles doc chunking (500 tokens w/ 100 overlap)


## 📌 Dev Tips

- To add support for `.docx` or `.pdf`, extend `VectorStore.legal_file_extensions` and add the loader.
- Vector DB is only populated if empty – smart init!
- Chat sessions are tracked using `phone_number` 🧾


## 📈 Possible Improvements

- Add session cleanup or expiry logic
- Dockerize for portability
- Add chat UI frontend (React/Vue)
- Expose admin logs via API
- Caching recent vector results

## Want to Learn More? 🤓

If you're passionate about **AI/ML**, **Node.js**, or **DevOps**, I’d love to connect and collaborate! Whether you're working on a project, need help with architecture, or just want to brainstorm cool tech stuff — I’m here to help! 💡✨

👉 **Follow me on Medium** for tutorials, guides, and deep dives into real-world tech problems. [Medium](https://prashant1879.medium.com/) 📚  
👉 **Need help with an AI/ML project?** Let’s talk! Reach out to me directly on [LinkedIn](https://www.linkedin.com/in/prashantsuthar1/). 🤖💬

Stay curious, stay awesome, and keep coding! 👨‍💻👩‍💻🚀