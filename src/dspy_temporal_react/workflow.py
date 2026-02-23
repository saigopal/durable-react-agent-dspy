from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from dspy_temporal_react.activities import (
        llm_plan_step,
        persist_result,
        synthesize_answer,
    )


@workflow.defn
class ReactAgentWorkflow:
    @workflow.run
    async def run(self, question: str) -> dict[str, Any]:
        scratchpad: list[dict[str, Any]] = []
        draft_answer = ""
        max_steps = 8

        for step_idx in range(max_steps):
            plan = await workflow.execute_activity(
                llm_plan_step,
                {"question": question, "scratchpad": scratchpad, "step": step_idx},
                start_to_close_timeout=timedelta(minutes=1),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            scratchpad.append(
                {
                    "phase": "plan",
                    "step": step_idx,
                    "thought": plan.get("thought", ""),
                    "plan": plan,
                }
            )

            if plan.get("type") == "final":
                draft_answer = str(plan.get("answer", "")).strip()
                break

            tool_name = str(plan.get("tool", "")).strip()
            tool_args = plan.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {"input": str(tool_args)}
            tool_result = await workflow.execute_activity(
                tool_name,
                tool_args,
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            scratchpad.append(
                {
                    "phase": "tool",
                    "step": step_idx,
                    "tool": tool_result.get("tool"),
                    "input": tool_result.get("input"),
                    "output": tool_result.get("output"),
                    "is_error": tool_result.get("is_error", False),
                }
            )

        result = await workflow.execute_activity(
            synthesize_answer,
            {
                "question": question,
                "scratchpad": scratchpad,
                "draft_answer": draft_answer,
            },
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        output_path = await workflow.execute_activity(
            persist_result,
            result,
            start_to_close_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return {"result": result, "artifact": output_path}
