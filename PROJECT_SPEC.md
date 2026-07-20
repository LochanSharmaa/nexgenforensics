# PROJECT_SPEC: SIFS Imagination Expander AI

## Product Vision
SIFS Imagination Expander AI (SIFS IE AI) is an AI-powered creative direction assistant for graphic designers. The app accepts a rough, messy creative idea ("imagination") from a user and expands it into 10 distinct, professional visual concept directions. Designers use these concepts for visual inspiration and execute their final artwork manually. 

### Key Rules
- **No Design Replacement**: Position AI as a brainstorming tool. The designer remains the final creator.
- **Reference-Only Labeling**: Every generated image and image prompt must display the exact notice: `"Reference only - final artwork belongs to the designer."`
- **Tagline**: `"One imagination. Ten creative directions."`

---

## Core Product Flow
1. **Rough Imagination Input**: User inputs a text description, design type, and optional mood preferences.
2. **Creative Brief Extraction**: The AI extracts a structured creative brief.
3. **Clarifying Questions**: The AI creates 5–8 clarifying questions. The user can answer or skip.
4. **10 Concept Directions**: The AI generates 10 visual concepts, each representing a unique style category (listed below).
5. **Diversity Analysis**: The system analyzes style/concept overlap, warning if two concepts are too similar (above a similarity index of 0.82) and offering a regeneration prompt.
6. **Actions**: Users can save/reject concepts, regenerate specific concepts, combine two concepts, or trigger reference-only image generations.
7. **Reference Images**: Generated on demand using the selected provider (mock, Gemini, ComfyUI, etc.) and marked clearly as "Reference only".
8. **Export**: Export the final concept board to JSON or Markdown formats.

---

## The 10 Concept Style Categories
Exactly one concept is generated for each of the following styles:
1. **Minimal luxury**
2. **Cinematic dramatic**
3. **Futuristic premium**
4. **Surreal dreamlike**
5. **Editorial magazine**
6. **Product advertisement**
7. **Abstract artistic**
8. **Dark premium**
9. **Human emotional story**
10. **Experimental poster**

---

## Technical Architecture

### Frontend
- Next.js (App Router, TypeScript)
- Tailwind CSS
- Framer Motion for premium UI animations
- shadcn/ui components

### Backend
- FastAPI (Python 3.11+)
- SQLModel / SQLAlchemy for database models
- SQLite (local development default) / PostgreSQL + pgvector (production)
- Redis + RQ (Redis Queue) for background image generation worker threads
- Cloudinary / S3-compatible or local uploads storage configuration

### AI Orchestration Layer
- **LLM Abstractions**: `MockLLMProvider`, `GeminiProvider`, `OpenAICompatibleLLMProvider`, `LocalVLLMProvider`.
- **Image Abstraction**: `MockImageProvider`, `GeminiImageProvider`, `ComfyUIProvider`, `DiffusersPlaceholderProvider`.
- **Quality Control**: JSON Schema parsing/validation, diversity matching logic, "Reference only" injection.
