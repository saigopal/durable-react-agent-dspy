from __future__ import annotations

import argparse
import asyncio
import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from dspy_temporal_react.workflow import ReactAgentWorkflow


async def _run(question: str, workflow_id: str | None) -> dict:
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "dspy-react-task-queue")

    client = await Client.connect(address, data_converter=pydantic_data_converter)

    result = await client.execute_workflow(
        ReactAgentWorkflow.run,
        question,
        id=workflow_id or f"react-agent-{uuid4()}",
        task_queue=task_queue,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy ReAct Temporal workflow")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--workflow-id", required=False, help="Optional workflow id")
    args = parser.parse_args()

    result = asyncio.run(_run(args.question, args.workflow_id))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
