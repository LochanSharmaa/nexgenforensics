<<<<<<< HEAD
# SIFS Imagination Expander AI (SIFS IE AI)

> AI-powered creative direction assistant for designers. One imagination. Ten creative directions.

SIFS IE AI is a professional brainstorming workspace for graphic designers. It expands a raw, unstructured idea ("imagination") into 10 highly distinct, professional visual directions, complete with reference image prompts, styling, background composition, and execution notes. All generated images are strictly labeled "Reference only".

---

## Technical Architecture

The app is built as a provider-agnostic monorepo:
- **Frontend**: Next.js (App Router), TypeScript, Tailwind CSS, Framer Motion, shadcn/ui.
- **Backend**: FastAPI (Python 3.11+), SQLModel/SQLAlchemy, SQLite (default for development) / PostgreSQL + pgvector, Redis (for background jobs).
- **AI Abstraction Layer**: Swappable LLM, Image, and Embedding providers (Mock, Gemini, OpenAI, Local vLLM, ComfyUI).

---

## Setup & Installation

### Prerequisite Checklist
- **Python**: version 3.11 or higher
- **Node.js**: version 18 or higher
- **Redis** & **PostgreSQL** (Optional, falls back to SQLite & in-memory/asyncio worker if Redis/postgres are not configured)

### 1. Backend Setup
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment variables template and customize it:
   ```bash
   cp ../.env.example .env
   ```
5. Run database migrations:
   ```bash
   alembic upgrade head
   ```
6. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js dev server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Model Selection Notes
- The default LLM provider uses `gemini-3.5-flash` for high context and fast, structured concept generation.
- The default Image provider uses `gemini-3.1-flash-image` for high-quality, reference-only image generations.
- A list of active Gemini model updates can be reviewed at the official [Gemini API Changelog](https://ai.google.dev/gemini-api/docs/changelog). Check there before relying on hardcoded model schemas long-term.

---

## License
MIT License. Reference only - final artwork belongs to the designer.
=======

>>>>>>> 4be368fcc5342fef2e8a88b1f5104b94ffb43690
