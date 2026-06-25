# Enterprise Agentic RAG Platform

A production-grade, enterprise-scale, cloud-native **Agentic RAG Platform** orchestrated with **LangGraph**, backed by a **FastAPI** service and a **Streamlit** user interface.

## 🚀 Key Features
- **LangGraph Orchestration**: Robust multi-node graph (Planner → Retriever → Reranker → Responder → Memory).
- **FastAPI Backend**: Asynchronous HTTP endpoints with API-key authentication, logging middleware, and rate-limiting.
- **Dual Streamlit UIs**:
  - **Chat Interface (Port 8501)**: Modern dark glassmorphic design, file uploader, citation trees, live metrics telemetry, and feedback collection.
  - **Evaluation Dashboard (Port 8502)**: Interactive RAGAS quality testing, historic run charts, and sample-level metric breakdowns.
- **Hybrid Vector Retrieval**: Qdrant combination of dense semantic similarity and sparse keyword search (BM25) with Reciprocal Rank Fusion (RRF) and FlashRank reranking.
- **Semantic Caching**: Automatic query response caching in Qdrant with age-based TTL expiration to cut down LLM latency and costs.
- **Conversational Memory**: PostgreSQL-backed persistent message history with automatic async summarization when chat history exceeds thresholds.
- **Input/Output Guardrails**: NeMo Guardrails safety checks (prompt injection, jailbreaks, PII sanitization, off-topic filtering).

---

## 🛠️ Environment Configuration

Copy the `.env.example` file to `.env` in the root directory:
```bash
cp .env.example .env
```

Configure the following variables in `.env`:
- `PORTKEY_API_KEY`: API Key for Portkey AI Gateway.
- `API_KEY`: Custom API token to secure FastAPI endpoints (default: `changeme`).
- `POSTGRES_PASSWORD`: PostgreSQL DB password.
- `OPENAI_API_KEY`: Optional OpenAI key.
- `BACKEND_URL`: URL configuration pointing stream apps to the backend (default: `http://localhost:8001/api/v1` to avoid conflicts).

---

## 🏃 Running the Platform

### Method 1: Local Development (Recommended)

To run the application services locally on your machine:

#### Step 1: Start Qdrant Vector Store
Ensure the Qdrant local binary is running:
```bash
./bin/qdrant
```
*Port: `6333` (HTTP) and `6334` (gRPC)*

#### Step 2: Start PostgreSQL Database
Ensure PostgreSQL is active and running locally on port `5432` with your credentials matching the `.env` settings.

#### Step 3: Run the FastAPI Backend
Start the backend server on port `8001` (to prevent port conflicts with other local mock servers on port 8000):
```bash
uvicorn src.api.main:create_app --factory --host 0.0.0.0 --port 8001
```

#### Step 4: Run the Streamlit Chat App
Launch the main user chat interface:
```bash
BACKEND_URL=http://localhost:8001/api/v1 streamlit run frontend/app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

#### Step 5: Run the Streamlit Evaluation App
Launch the developer evaluation dashboard:
```bash
BACKEND_URL=http://localhost:8001/api/v1 streamlit run frontend/eval_app.py --server.port 8502
```
Open `http://localhost:8502` in your browser.

---

### Method 2: Docker Compose (All-in-One Container)

To orchestrate and run the full stack (API, Frontend, Eval UI, PostgreSQL, Qdrant) inside isolated Docker containers:

```bash
docker-compose up --build
```

- **FastAPI REST API Docs**: `http://localhost:8000/docs`
- **Streamlit Chat Application**: `http://localhost:8501`
- **Evaluation Dashboard**: `http://localhost:8502`

---

## 💡 How to Use the Chat Interface

1. **Upload Reference Documents**:
   Use the sidebar file uploader to load any `.pdf`, `.docx`, `.pptx`, `.html`, or `.txt` reference document.
2. **Process & Ingest**:
   Click **🚀 Process & Ingest Document**. The system will split the document into overlapping chunks, embed them, index them in Qdrant, and **automatically clear any previously ingested documents and stale caches**.
3. **Chat**:
   Ask questions about the uploaded file (e.g. *"What are the core phases outlined in the PDF?"*).
4. **Citations & Metrics**:
   Expand **📚 View Document Citations** below responses to inspect the source filename, page number, and similarity score. Check operational stats like Latency and Cache hit rate in the sidebar.

---

## 🧪 Running Evaluations (RAGAS)

1. Open the Evaluation UI (`http://localhost:8502`).
2. Select your golden dataset (e.g. `golden_qa`).
3. Choose the evaluation metrics you'd like to measure (Faithfulness, Answer Relevancy, Context Recall, Context Precision).
4. Click **Run Evaluation** to trigger the RAGAS evaluation pipeline.
5. Review the radar scores, tabular results, and click on any row to inspect context retrieval performance at a granular level.
