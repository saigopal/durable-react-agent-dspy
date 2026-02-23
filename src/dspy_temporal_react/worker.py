from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from dspy_temporal_react.activities import (
    dynamic_tool_activity,
    llm_plan_step,
    persist_result,
    synthesize_answer,
)
from dspy_temporal_react.workflow import ReactAgentWorkflow


async def _run_worker() -> None:
    load_dotenv()

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "dspy-react-task-queue")

    client = await Client.connect(address, data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ReactAgentWorkflow],
        activities=[llm_plan_step, dynamic_tool_activity, synthesize_answer, persist_result],
    )
    await worker.run()


def main() -> None:
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
