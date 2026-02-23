from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import dspy
import httpx
from dspy.adapters.types.tool import Tool
from dspy.signatures.signature import ensure_signature
from litellm import ContextWindowExceededError
from temporalio import activity
from temporalio.common import RawValue

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _build_lm() -> dspy.LM:
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
    # Keep SDK-level retries at 1 attempt so Temporal retry policy is the
    # single source of truth for backoff/retry behavior.
    return dspy.LM(model_name, api_key=os.getenv("OPENAI_API_KEY"), num_retries=0)


AGENT_SIGNATURE = ensure_signature("question -> answer")


async def get_ip_address() -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()


async def finish() -> str:
    """Signals that the agent has enough information to finalize the answer."""
    return "Completed."


TOOL_REGISTRY = {
    "get_ip_address": get_ip_address,
    "finish": finish,
}


def get_handler(tool_name: str):
    """Return the async handler for a given tool name."""
    handler = TOOL_REGISTRY.get(tool_name)
    if handler is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return handler


def _build_instruction(signature: dspy.Signature, tools: dict[str, Tool]) -> str:
    inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
    outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
    instr = [f"{signature.instructions}\n"] if signature.instructions else []

    instr.extend(
        [
            f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
            f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
            "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
            "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
            "When writing next_thought, you may reason about the current situation and plan for future steps.",
            "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
        ]
    )
    for idx, tool in enumerate(tools.values()):
        instr.append(f"({idx + 1}) {tool}")
    instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")
    return "\n".join(instr)


@lru_cache(maxsize=1)
def _build_react_modules() -> tuple[dict[str, Tool], Any, Any]:
    tools = {"get_ip_address": Tool(get_ip_address)}
    tools["finish"] = Tool(
        func=finish,
        name="finish",
        desc="Marks the task as complete and signals that all required output fields are available.",
        args={},
    )

    react_signature = (
        dspy.Signature({**AGENT_SIGNATURE.input_fields}, _build_instruction(AGENT_SIGNATURE, tools))
        .append("trajectory", dspy.InputField(), type_=str)
        .append("next_thought", dspy.OutputField(), type_=str)
        .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
        .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
    )
    fallback_signature = dspy.Signature(
        {**AGENT_SIGNATURE.input_fields, **AGENT_SIGNATURE.output_fields},
        AGENT_SIGNATURE.instructions,
    ).append("trajectory", dspy.InputField(), type_=str)

    return tools, dspy.Predict(react_signature), dspy.ChainOfThought(fallback_signature)


def _scratchpad_to_trajectory(scratchpad: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory: dict[str, Any] = {}
    for step in scratchpad:
        idx = step.get("step")
        if not isinstance(idx, int):
            continue
        phase = step.get("phase")
        if phase == "plan":
            plan = step.get("plan", {})
            trajectory[f"thought_{idx}"] = str(step.get("thought", ""))
            if plan.get("type") == "tool":
                trajectory[f"tool_name_{idx}"] = str(plan.get("tool", ""))
                args = plan.get("args", {})
                trajectory[f"tool_args_{idx}"] = args if isinstance(args, dict) else _to_dict_like(args)
            elif plan.get("type") == "final":
                trajectory[f"tool_name_{idx}"] = "finish"
                trajectory[f"tool_args_{idx}"] = {}
        elif phase == "tool":
            trajectory[f"observation_{idx}"] = step.get("output", "")
    return trajectory


def _format_trajectory(trajectory: dict[str, Any]) -> str:
    if not trajectory:
        return "No trajectory yet."
    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
    return adapter.format_user_message_content(trajectory_signature, trajectory)


def _truncate_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Drop the oldest thought/tool/args/observation bundle."""
    keys = list(trajectory.keys())
    if len(keys) < 4:
        raise ValueError("Trajectory too short to truncate further")
    for key in keys[:4]:
        trajectory.pop(key, None)
    return trajectory


def _normalize_tool_args(step: dict[str, Any]) -> dict[str, Any]:
    """Normalize planner tool args while preserving back-compat."""
    raw_args = step.get("args")
    if isinstance(raw_args, dict):
        return raw_args

    # Back-compat with older planner shape: {"input": "..."}
    tool_input = step.get("input")
    if tool_input is not None:
        return {"input": tool_input}

    return {}


def _to_dict_like(value: Any) -> dict[str, Any]:
    """Best-effort normalization to dict for model-produced tool args."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"input": value}
    if value is None:
        return {}
    return {"input": str(value)}


def _normalize_tool_name(tool_name: str) -> str:
    """Map common aliases to canonical tool names."""
    key = tool_name.strip().lower()
    aliases = {
        "ip": "get_ip_address",
        "get_ip": "get_ip_address",
        "ip_address": "get_ip_address",
        "final": "finish",
        "done": "finish",
    }
    return aliases.get(key, key)


def _tool_requires_args(tool_name: str) -> bool:
    """Return whether the tool handler requires arguments."""
    handler = TOOL_REGISTRY.get(tool_name)
    if handler is None:
        return True
    return len(inspect.signature(handler).parameters) > 0


@activity.defn
async def llm_plan_step(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload["question"])
    scratchpad = payload.get("scratchpad", [])
    if not isinstance(scratchpad, list):
        scratchpad = []
    trajectory = _scratchpad_to_trajectory(scratchpad)
    _, react_predict, _ = _build_react_modules()

    # Retry planner call with trajectory truncation if context window is exceeded.
    for _ in range(3):
        try:
            with dspy.context(lm=_build_lm()):
                pred = react_predict(
                    question=question,
                    trajectory=_format_trajectory(trajectory),
                )
            break
        except ContextWindowExceededError:
            try:
                logger.warning("Trajectory exceeded context window; truncating oldest tool call bundle.")
                trajectory = _truncate_trajectory(trajectory)
            except ValueError:
                return {
                    "type": "final",
                    "thought": "Context window exceeded; trajectory cannot be truncated further.",
                    "answer": "I could not continue because context is too large.",
                }
        except ValueError as err:
            logger.warning("Ending trajectory due to invalid tool selection: %s", _fmt_exc(err))
            return {
                "type": "final",
                "thought": "Agent failed to select a valid tool.",
                "answer": "I could not select a valid next step. Please clarify your question.",
            }
    else:
        return {
            "type": "final",
            "thought": "Context window exceeded repeatedly; finalizing.",
            "answer": "I could not continue reasoning because the context window was exceeded.",
                }

    thought = str(getattr(pred, "next_thought", "")).strip()
    tool_name = _normalize_tool_name(str(getattr(pred, "next_tool_name", "")).strip())
    raw_args = getattr(pred, "next_tool_args", {})
    tool_args = _to_dict_like(raw_args)

    if tool_name == "finish":
        return {
            "type": "final",
            "thought": thought,
            "answer": "",
        }

    # Back-compat if planner still emits legacy shape in args.
    tool_args = _normalize_tool_args({"args": tool_args, "input": tool_args.get("input")})
    if tool_name not in TOOL_REGISTRY:
        return {
            "type": "final",
            "thought": "Tool request invalid; finishing safely.",
            "answer": "I could not select a valid tool step. Please clarify your question.",
        }
    if _tool_requires_args(tool_name) and not tool_args:
        return {
            "type": "final",
            "thought": "Tool request invalid; finishing safely.",
            "answer": "I could not select a valid tool step. Please clarify your question.",
        }
    return {
        "type": "tool",
        "thought": thought,
        "tool": tool_name,
        "args": tool_args,
    }


@activity.defn(dynamic=True)
async def dynamic_tool_activity(args: Sequence[RawValue]) -> dict[str, Any]:
    """Execute a tool dynamically based on activity name."""
    tool_name = activity.info().activity_type

    if not args:
        return {
            "tool": tool_name,
            "input": {},
            "output": "Tool error: missing input payload",
            "is_error": True,
        }

    tool_args = activity.payload_converter().from_payload(args[0].payload, dict)
    activity.logger.info("Running dynamic tool '%s' with args: %s", tool_name, tool_args)

    try:
        handler = get_handler(tool_name)
        if not inspect.iscoroutinefunction(handler):
            raise TypeError("Tool handler must be async (awaitable).")

        sig = inspect.signature(handler)
        params = list(sig.parameters.values())

        if len(params) == 0:
            result = await handler()
        else:
            param = params[0]
            param_name = param.name
            ann = param.annotation

            if isinstance(ann, type) and issubclass(ann, BaseModel):
                nested_args = tool_args.get(param_name, tool_args)
                result = await handler(ann(**nested_args))
            elif len(params) == 1 and param_name not in tool_args and "input" in tool_args:
                # Supports current planner shape: {"input": "..."}.
                result = await handler(tool_args["input"])
            else:
                result = await handler(**tool_args)

        activity.logger.info("Tool '%s' result: %s", tool_name, result)
        return {
            "tool": tool_name,
            "input": tool_args,
            "output": result,
            "is_error": False,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "tool": tool_name,
            "input": tool_args,
            "output": f"Execution error in {tool_name}: {_fmt_exc(exc)}",
            "is_error": True,
        }


@activity.defn
async def synthesize_answer(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload["question"])
    scratchpad = payload.get("scratchpad", [])
    draft_answer = str(payload.get("draft_answer", "")).strip()
    if not isinstance(scratchpad, list):
        scratchpad = []
    trajectory = _scratchpad_to_trajectory(scratchpad)

    _, _, fallback_extract = _build_react_modules()
    pred = None
    for _ in range(3):
        try:
            with dspy.context(lm=_build_lm()):
                pred = fallback_extract(
                    question=question,
                    trajectory=_format_trajectory(trajectory),
                )
            break
        except ContextWindowExceededError:
            try:
                logger.warning("Fallback extraction exceeded context window; truncating trajectory.")
                trajectory = _truncate_trajectory(trajectory)
            except ValueError:
                break
        except ValueError as err:
            logger.warning("Fallback extraction ended due to invalid trajectory truncation: %s", _fmt_exc(err))
            break

    if pred is not None:
        answer = str(getattr(pred, "answer", "")).strip() or draft_answer or "No answer generated."
    else:
        answer = draft_answer or "No answer generated."

    info = activity.info()
    return {
        "question": question,
        "answer": answer,
        "run_id": info.workflow_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace": scratchpad,
    }


@activity.defn
async def persist_result(result: dict[str, Any]) -> str:
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(result.get("run_id", "unknown-run"))
    output_path = runs_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(output_path)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """Return one-string traceback summary for tool failures."""
    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
