# Multi-Agent Project Planner 📐🤖

Welcome to the **Autonomous Project Planning & Estimation Suite**. 

This system coordinates three specialized AI agents to analyze project requirements, estimate task complexity, and allocate resources efficiently.

---

### 🛠️ System Architecture

*   **Project Planner Agent**: Analyzes constraints and requirements to produce a structured task list breakdown.
*   **Estimation Analyst Agent**: Computes duration, resource hours, and potential execution risks.
*   **Allocation Strategist Agent**: Maps tasks to team members' skillsets, optimizing the project schedule.

### 🚀 Execution Pipeline

1.  **LLM Configuration**: Select your LLM backend (OpenAI, Groq, or local Ollama).
2.  **Knowledge Upload (Optional)**: Provide past project documents to ground estimations using local vector embeddings.
3.  **Requirement Collection**: Input project type, objectives, team list, and constraints.
4.  **Task Approval Loop**: Interactively inspect, modify, and approve the generated task breakdown.
5.  **Dashboard Generation**: Output a complete, styled HTML project timeline, milestone list, and resource map.
