# DSPy ReAct Agent on Temporal

Durable ReAct agent implementation using:
- Temporal **workflow** for deterministic orchestration
- Temporal **activities** for non-deterministic work
- DSPy for planner + final extraction signatures

## Architecture

- `src/dspy_temporal_react/workflow.py`
  - Durable ReAct loop (`max_steps`, activity sequencing, retries)
- `src/dspy_temporal_react/activities.py`
  - `llm_plan_step`
  - `dynamic_tool_activity` (`@activity.defn(dynamic=True)`)
  - `synthesize_answer`
  - `persist_result`
- `src/dspy_temporal_react/worker.py`
  - Worker registration
- `src/dspy_temporal_react/client.py`
  - CLI runner for workflow execution

Execution flow:
`llm_plan_step -> dynamic_tool_activity -> repeat -> synthesize_answer -> persist_result`

## Tools

Current tool registry in `src/dspy_temporal_react/activities.py`:
- `get_ip_address()`
- `get_weather(city: str)` (with provider fallback)
- `calculator(expression: str)` (safe AST evaluator, supports `+ - * / %` and parentheses)
- `python_sandbox(code: str)` (bounded subprocess execution)
- `finish()` (always available completion tool)

## Prerequisites

1. Python `>=3.10`
2. Docker (recommended for local Temporal)
3. OpenAI-compatible API key for DSPy LM calls

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Temporal locally

```bash
docker compose up -d
```

Temporal endpoints:
- gRPC: `localhost:7233`
- UI: `http://localhost:8080`

## Environment

Create `.env`:

```bash
OPENAI_API_KEY=your_key_here
MODEL_NAME=openai/gpt-4o-mini
TEMPORAL_ADDRESS=localhost:7233
TEMPORAL_TASK_QUEUE=dspy-react-task-queue
```

Optional tuning knobs:

```bash
MAX_OBSERVATION_CHARS=1200
PY_SANDBOX_TIMEOUT_SECONDS=5
PY_SANDBOX_MAX_OUTPUT_CHARS=4000
```

## Run

Start worker:

```bash
react-worker
```

Run client in another terminal:

```bash
react-run --question "What is my public IP address?"
```

Weather/calculation example:

```bash
react-run --question "If tomorrow’s temperature (in °C) in bangalore is a prime number, multiply it by 7 and subtract today’s low temperature. If it is not prime, divide today’s high temperature by 2 and add 11. Use tools only when necessary."
```

## Output

- Workflow result is printed by client as JSON.
- Full run artifact is written by `persist_result` to:
  - `runs/<workflow_run_id>.json`

## Notes

- Workflow code remains deterministic; all side effects are in activities.
- Planner and extractor use DSPy-style signatures with trajectory formatting/truncation.
- Tool dispatch is name-based through one dynamic activity boundary.
- LM is scoped per activity via `dspy.context(...)`; SDK retries are disabled (`num_retries=0`) so Temporal owns retries.
