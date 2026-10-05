# AODLF Project (v0.1.0)

An adaptive, modular AI educational platform designed to bridge structured academic instruction with open-ended personal exploration through a highly efficient, intelligent architecture.



Goal is to build a resilient, intelligent, and token-efficient conversational AI framework.
## Achieved Milestones
We have completed the core components required for semantic orchestration:

*   **Router Refactor:** The `IntelligentRouter` has been refactored to support asynchronous routing, enabling LLM calls (`_semantic_classify`) instead of relying solely on fast keywords. It also incorporates a critical "Heuristic Fallback" mechanism for robustness.
*   **Context Manager Upgrade:** The `ContextManager` is upgraded from simple string truncation to **Semantic Summarization**. It now uses the active provider's LLM capabilities to summarize older conversation history, preserving long-term intent and context while managing token budget effectively.
*   **Orchestrator Synchronization:** The `BackendOrchestrator` has been updated to run all critical functions (`route_prompt`, `check_and_compact`) asynchronously, linking the new components into a cohesive, modern pipeline.

## Incomplete Work & Immediate Goals (Roadmap)
The system is functional at an integrated level, but ongoing work focuses on refinement and stability:

*   **End-to-End Verification:** Rigorously testing the entire chain (Orchestrator $\rightarrow$ Router $\rightarrow$ Provider).
*   **Latency Optimization:** Fine-tuning the "Grey Area" logic in the router to ensure fast heuristics are prioritized over expensive semantic calls when possible, keeping latency low.
*   **Model Fallback Hardening:** Ensuring the system gracefully degrades if the Semantic Classification LLM fails or times out.