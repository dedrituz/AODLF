# AODLF — Automated Open-Domain Learning Framework

> **v0.3.3** · Python · FastAPI · Ollama · Multi-Provider AI Backend

An adaptive, modular AI orchestration backend designed to evolve into a full educational platform. At its current stage, it provides a sophisticated general-purpose AI chat backend with intelligent tiered routing, multimodal document processing, and support for both local and cloud inference providers.

---

## Overview

AODLF (Automated Open-Domain Learning Framework) is built around the idea that an AI assistant for learning should be *intelligent about resources*: routing simple questions to fast models, complex reasoning to powerful ones, and visual content through whichever pipeline is most efficient. The backend is designed from the ground up to eventually support two distinct educational modes — a RAG-restricted **Classroom Mode** for structured course study, and an open-domain **Exploration Mode** for personal research — but the current implementation focuses on the core inference and routing infrastructure that underpins both.

---

## Current Features

- **Intelligent Tiered Routing** — Combines fast keyword heuristics with asynchronous LLM-based semantic classification to automatically route queries to the most appropriate model tier (basic, complex, or vision), balancing speed and capability.
- **Multimodal Document Intelligence** — Supports image and document attachments. Images are analyzed for text density and routed through either a lightweight OCR pipeline (PaddleOCR → EasyOCR → Tesseract) or a vision model. PDFs are parsed with a hybrid approach: digital text extraction for standard pages, OCR-based transcription for scanned pages.
- **Semantic Context Compaction** — Long conversations are automatically summarized by the active LLM (rather than simply truncated) to preserve meaning while managing token budgets.
- **Hardware-Aware Model Assignment** — Scans local CPU, RAM, and GPU/VRAM to intelligently suggest which models to assign to each routing tier.
- **Hybrid Inference** — Supports local models via Ollama (and any custom OpenAI-compatible endpoint such as llama.cpp or vLLM) alongside major cloud providers.
- **Real-Time SSE Streaming** — Multi-channel streaming via Server-Sent Events: separate events for status updates, thinking/reasoning traces, response tokens, and final metadata.
- **Post-Response Intelligence** — After each response, a lightweight model generates 3–4 contextually relevant follow-up questions, and the session is auto-titled on the first turn.
- **Dynamic Configuration** — API keys and provider endpoints can be added or changed live from within the CLI, without restarting.

---

## Supported Providers

| Provider | Type | Notes |
|----------|------|-------|
| Ollama | Local | Default; no API key required |
| Custom (llama.cpp / vLLM / etc.) | Local | Any OpenAI-compatible endpoint |
| OpenAI | Cloud | GPT-4o, GPT-4o-mini, o3-mini |
| Groq | Cloud | Ultra-low latency LPU inference |
| DeepSeek | Cloud | DeepSeek-V3, DeepSeek-R1 (with thinking traces) |
| Anthropic | Cloud | Claude 3.5 Sonnet, Haiku, Opus |
| OpenRouter | Cloud | Gateway to 200+ models |

---

## Usage

### Quick Start

**Windows:**
```bat
run.bat
```

**Linux / macOS:**
```bash
./run.sh
```

> If startup takes a while, it is likely that PaddleOCR or EasyOCR are being loaded into memory on first run. This is a one-time initialization cost per session.

---

### CLI Mode (Default)

Launches the interactive terminal interface. This is the primary way to use AODLF.

```bash
python main.py
```

The CLI walks you through a 5-step setup on each launch:

1. **Initialization** — Scans all configured providers and tests connectivity.
2. **Provider Selection** — Choose from any connected local or cloud provider.
3. **Model Routing Setup** — Review the auto-suggested tier assignments (Basic / Complex / Vision). Accept, customize per-tier, or select a single fixed model.
4. **Model Warmup** — Verifies the model is loaded and ready (with a live timer spinner).
5. **Chat** — Stream responses with live status updates, optional thinking traces, and suggested follow-ups after each turn.

#### Attaching Files

Paste a full absolute path (quoted or unquoted) anywhere in your message:

```
You: Summarize this document "C:\Users\me\Documents\lecture_notes.pdf"
You: What does this diagram show? C:\Users\me\Desktop\circuit.png
```

---

### API Server Mode

Runs a FastAPI backend for integration with web or mobile frontends.

```bash
python main.py --server --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service availability and version check |
| `/api/providers` | GET | All provider statuses and model counts |
| `/api/models` | GET | Model list with smart routing tier assignments |
| `/api/specs` | GET | Local hardware specifications |
| `/api/config/api-key` | POST | Dynamically update an API key or endpoint URL |
| `/api/compact` | POST | Manually trigger semantic context compaction |
| `/api/chat` | POST | Core chat endpoint — streams via SSE |

---

### CLI Commands

Type any command at the prompt during a session:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands for the current mode |
| `/specs` | Display scanned hardware specs and VRAM/RAM details |
| `/debug` | Inspect internal state: session info, token counts, tier assignments |
| `/route` | View the current smart routing tier configuration |
| `/model <name>` | Override the active model for all subsequent prompts |
| `/model auto` | Reset model override and return to smart routing |
| `/compact` | Manually trigger context compaction |
| `/clear` | Clear conversation history for the current session |
| `/add-api <provider> <key>` | Add or update an API key |
| `/set-endpoint <provider> <url>` | Set a custom base URL for a provider |
| `/list` | List providers (in menu) or models with tier tags (in chat) |
| `/back` | Return to provider selection |
| `/exit` | Exit the application |

---

## Configuration

Create a `.env` file in the project root to configure providers:

```env
# Local providers (optional — Ollama defaults to http://localhost:11434)
OLLAMA_BASE_URL="http://localhost:11434"
CUSTOM_ENDPOINT_URL="http://localhost:8080/v1"

# Cloud providers (add any you want to use)
OPENAI_API_KEY="sk-..."
GROQ_API_KEY="gsk_..."
DEEPSEEK_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
OPENROUTER_API_KEY="sk-or-..."
```

Keys can also be added or updated live using `/add-api` or the `/api/config/api-key` endpoint — no restart required.

---

## Installation

```bash
pip install -r requirements.txt
```

> For OCR support, at least one of PaddleOCR, EasyOCR, or Tesseract must be available. The system gracefully falls back through each in order.

---

## Project Status

AODLF is in active early development. The backend infrastructure layer is largely complete. The platform-level educational features (RAG, Classroom Mode, workspaces, Vault) are planned for future development phases.

### What's Working
- Full inference pipeline with tiered routing and streaming
- Multimodal image and document processing
- Semantic context management and compaction
- All 7 provider integrations (local + cloud)
- Complete CLI interface and FastAPI server

### Immediate Next Steps
- Fix `/compact` command (async bug in CLI and API)
- Wire session persistence (`ChatStore` is scaffolded but not yet connected)
- Build RAG engine (the first step toward Classroom Mode)
- Develop a minimal web frontend for the full streaming UX

### Eventual Vision
The platform is designed to evolve into a full AI learning environment with two modes:

- **Classroom Mode** — RAG-restricted responses from uploaded syllabus and course materials, ensuring factual accuracy within a defined knowledge boundary.
- **Exploration Mode** — Unrestricted open-domain inquiry powered by external MCP tools (GitHub, StackOverflow, scholarly databases).

Supporting both modes will be a **Practice Workspace** (quizzes, flashcards), an **Exploration Workspace**, and a **Vault** — a personal hub for organizing research notes and revision materials.

For a full breakdown of what is implemented, how each module works, known bugs, and the complete development roadmap, see [`implementation_tracker.md`](implementation_tracker.md).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend Framework | FastAPI + Uvicorn |
| CLI Interface | Rich |
| HTTP Client | HTTPX (async) |
| OCR (Primary) | PaddleOCR + PaddlePaddle |
| OCR (Secondary) | EasyOCR |
| OCR (Fallback) | Pytesseract |
| Image Processing | Pillow, OpenCV, NumPy |
| PDF Parsing | pypdf + pypdfium2 |
| GPU Detection | PyTorch CUDA / nvidia-smi |
| Streaming Protocol | Server-Sent Events (SSE) via sse-starlette |
