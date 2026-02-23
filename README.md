# DSPy ReAct Agent on Temporal

This project wires a DSPy `ReAct` agent into Temporal using:
- A **workflow** for durable orchestration and retries
- **activities** for non-deterministic work (LLM planning, dynamic tool calls, answer synthesis, and persistence)

## What gets created
- `src/dspy_temporal_react/workflow.py`: Temporal workflow definition
- `src/dspy_temporal_react/activities.py`: step-planning LLM, dynamic tool activity, final-answer LLM, persistence
- `src/dspy_temporal_react/worker.py`: Temporal worker bootstrap
- `src/dspy_temporal_react/client.py`: starts a workflow run

## Prerequisites
1. Python 3.10+
2. Temporal server running locally (`localhost:7233`)
3. LLM key (example below uses OpenAI-compatible key via DSPy)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional: start Temporal locally via Docker:
```bash
docker compose up -d
```

Create `.env`:
```bash
OPENAI_API_KEY=your_key_here
MODEL_NAME=openai/gpt-4o-mini
TEMPORAL_ADDRESS=localhost:7233
TEMPORAL_TASK_QUEUE=dspy-react-task-queue
```

## Run
Start worker:
```bash
react-worker
```

In another terminal, run a request:
```bash
react-run --question "What is my public IP address?"
```

## Notes
- The workflow is deterministic and runs the ReAct loop.
- Each loop turn uses separate activities: `llm_plan_step` then a dynamic tool activity.
- Final response generation is a separate `synthesize_answer` activity.
- Planner tool steps now use structured args (`{\"args\": {...}}`) instead of only raw string input.
- Available callable tool is `get_ip_address` (plus internal `finish` signal).
