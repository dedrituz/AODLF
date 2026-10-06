# AODLF Project (v0.3.2)

*Adaptive Open-Domain Learning Framework*

An adaptive, modular AI orchestration framework designed to bridge structured academic instruction with open-ended personal exploration through intelligent architecture. It transitions from simple keyword matching to **Semantic Orchestration**—using LLMs for deep classification and context summarization.

## Core Capabilities & Intelligence

AODLF goes beyond standard chat wrappers by providing:
*   **Intelligent Tiered Routing:** Combines fast heuristics with asynchronous LLM-based semantic classification (`_semantic_classify`) to route queries to the most efficient model for your specific task, balancing speed and intelligence.
*   **Semantic Context Management:** Unlike simple truncation, AODLF uses **Semantic Summarization**. It leverages the active provider's LLM to summarize older conversation history, preserving long-term intent while managing token budgets effectively.
*   **Multimodal Intelligence:** Automatically detects input types. For images, it evaluates text density to decide whether to perform OCR or route to a dedicated Vision model. OCR is now functional with accurate score calculation, allowing the use of text models even when the context contains images.
*   **Hardware-Awareness:** The system scans your local hardware (CPU/GPU/VRAM) and uses that data to intelligently assign model tiers for optimal performance on your specific machine.
*   **Quiet Operation:** Logging and warnings from dependencies like PaddleOCR, PaddlePaddle, Padlex, and Torch are suppressed for a cleaner experience.

## Usage Modes

**Quick Launch:** Use `run.bat` or `run.sh` to initialise the CLI-based chat interface.
**Note:** If initialisation is taking time, it probably means that PaddleOCR or EasyOCR are being loaded.

### 1. Interactive CLI Mode
Best for rapid prototyping, research, and direct experimentation within the terminal.
*   **Launch:** `python main.py` (Default mode)

### 2. API Server Mode
Designed to act as a robust backend for web interfaces or mobile applications via high-performance SSE streaming.
*   **Launch:** `python main.py --server --host 127.0.0.1 --port 8000`

## API Reference

Available endpoints:

| Endpoint              | Method | Description                                                                                                |
| :-------------------- | :----: | :--------------------------------------------------------------------------------------------------------- |
| `/health`             | `GET`  | Check service availability and version.                                                                    |
| `/api/providers`      | `GET`  | Scan and list all configured providers, connection states, and available models.                           |
| `/api/models`         | `GET`  | Get a list of models with **Smart Routing** intelligence (tier assignment based on hardware & capability). |
| `/api/specs`          | `GET`  | Retrieve local system hardware specifications used for routing decisions.                                  |
| `/api/config/api-key` | `POST` | Dynamically update API keys for cloud providers.                                                           |
| `/api/compact`        | `POST` | Manually trigger semantic context compaction to save tokens.                                               |
| `/api/chat`           | `POST` | The core conversational endpoint supporting text and multi-modal inputs via SSE streaming.                 |

## Roadmap

*   **Latency Optimization:** Refining "Grey Area" logic for faster heuristic transitions.
*   **Model Fallback Hardening:** Enhancing graceful degradation if semantic models time out.
*   **End-to-End Verification:** Continuous testing of the Orchestrator -> Router -> Provider pipeline.
