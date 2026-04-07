#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.llm_engine import LLMEngine
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder

DEFAULT_RESULTS_DIR = "./results/260403-validation-gpt-5-azure"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_PROVIDER = "azure"
DEFAULT_ENDPOINT = "https://societalllm.openai.azure.com/"
DEFAULT_OUTPUT_DIR_NAME = "failure_analysis"
DEFAULT_MAX_TEXT_CHARS = 20000

ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]+)"')
RUN_NUMBER_RE = re.compile(r"_run_(\d+)(?:_|$)")
TASK_RECEIVED_RE = re.compile(
    r"\[TASK\]:\s*\nReceived at:\s*(.*?)\nSender:\s*(.*?)\nMessage:\s*(.*)",
    re.DOTALL,
)

TRACE_ANALYSIS_SYSTEM_PROMPT = textwrap.dedent(
    """
    You analyze failed agent trajectories from a benchmark environment.

    Your job is to infer why the agent failed from the execution trace itself.
    Do not merely restate the validation rationale. The rationale only states why
    the oracle comparison failed; you must explain the trajectory-level failure mode.

    Focus on issues such as:
    - omitted required steps
    - misunderstanding constraints
    - premature stopping
    - failed delegation / not resuming after delegation
    - wrong filtering or state tracking
    - missing monitoring or follow-up
    - lack of verification
    - wrong tool choice or wrong sequencing

    When a sub-agent completed a subtask but the main agent failed to continue,
    call that out explicitly.

    Return exactly one JSON object with this schema:
    {
      "root_cause_summary": "short paragraph",
      "failure_modes": ["snake_case_label"],
      "what_agent_missed": ["specific missed step"],
      "what_agent_got_wrong": ["specific misunderstanding or bad assumption"],
      "trajectory_based_explanation": "concise but concrete explanation grounded in the trace",
      "evidence": [
        {
          "agent": "Agent N",
          "message_index": 0,
          "action": "tool_or_final_action_or_null",
          "quote": "short quote or trace excerpt",
          "why_it_matters": "why this shows the failure"
        }
      ],
      "suggested_fix": ["specific change that would likely prevent this failure"],
      "confidence": "high|medium|low"
    }
    """
).strip()

AGGREGATE_SYSTEM_PROMPT = textwrap.dedent(
    """
    You aggregate structured failure analyses for agent trajectories.

    Produce exactly one JSON object with this schema:
    {
      "dataset_summary": "2-4 sentence summary of the main failure patterns",
      "top_failure_modes": [
        {
          "mode": "snake_case_label",
          "count": 0,
          "summary": "what this mode looked like in the traces"
        }
      ],
      "scenario_patterns": [
        {
          "scenario_id": "scenario id",
          "runs": [1, 2, 3],
          "common_failure": "what repeatedly went wrong"
        }
      ],
      "recommended_agent_improvements": [
        "concrete engineering improvement"
      ],
      "notable_examples": [
        {
          "scenario_id": "scenario id",
          "run_number": 1,
          "why_representative": "why this trace is representative"
        }
      ]
    }
    """
).strip()

logger = logging.getLogger("analyze_failure_traces")


@dataclass
class TraceRecord:
    trace_path: Path
    scenario_id: str
    run_number: int | None
    validation_decision: str | None
    validation_rationale: str | None
    interaction_histories: dict[str, list[dict[str, Any]]]
    llm_usage_stats: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an LLM to analyze failed benchmark traces and summarize why the "
            "agent failed from its trajectory, not just from validation_rationale."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="Directory containing benchmark outputs with a lite/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write analysis outputs. Defaults to <results-dir>/failure_analysis.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use for analysis.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="Provider to use for analysis.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Endpoint to use for analysis.",
    )
    parser.add_argument(
        "--api-version",
        default=None,
        help="Optional Azure API version. If provided, it is exported as AZURE_API_VERSION.",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Limit the number of traces analyzed.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help="Maximum characters to keep for each trace content block.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=8,
        help="Maximum number of per-trace analysis requests to run concurrently.",
    )
    parser.add_argument(
        "--include-valid",
        action="store_true",
        help="Include non-failed traces as well. By default only invalid traces are analyzed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute per-trace analyses even if cached outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare normalized trace inputs locally; do not call the LLM.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def shorten(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def parse_run_number(trace_path: Path) -> int | None:
    match = RUN_NUMBER_RE.search(trace_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def extract_action_name(content: str | None) -> str | None:
    if not content:
        return None
    match = ACTION_RE.search(content)
    if match is None:
        return None
    return match.group(1)


def extract_observation_payload(content: str | None) -> str:
    if not content:
        return ""
    text = content.strip()
    if "***" not in text:
        return text
    parts = text.split("***")
    if len(parts) >= 3:
        return parts[1].strip()
    return text


def extract_task(content: str | None) -> str:
    if not content:
        return ""
    match = TASK_RECEIVED_RE.match(content)
    if match is None:
        return content.strip()
    received_at, sender, message = match.groups()
    return f"Received at: {received_at}\nSender: {sender}\nMessage: {message.strip()}"


def summarize_tool_counts(history: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for message in history:
        if message.get("role") != "assistant":
            continue
        action_name = extract_action_name(message.get("content"))
        if action_name is None:
            continue
        counts[action_name] += 1
    return dict(sorted(counts.items()))


def render_history(
    history: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for index, message in enumerate(history):
        role = message.get("role")
        content = message.get("content") or ""

        if role == "system":
            continue

        item: dict[str, Any] = {
            "message_index": index,
            "role": role,
        }

        if role == "user":
            item["content"] = shorten(extract_task(content), max_chars)
        elif role == "assistant":
            item["action"] = extract_action_name(content)
            item["content"] = shorten(content, max_chars)
        elif role == "tool-response":
            item["content"] = shorten(extract_observation_payload(content), max_chars)
        else:
            item["content"] = shorten(content, max_chars)

        rendered.append(item)
    return rendered


def normalize_agent_label(index: int) -> str:
    return f"Agent {index + 1}"


def load_trace_records(
    results_dir: Path,
    include_valid: bool,
) -> list[TraceRecord]:
    lite_dir = results_dir / "lite"
    if not lite_dir.exists():
        raise FileNotFoundError(f"Missing lite trace directory: {lite_dir}")

    records: list[TraceRecord] = []
    for trace_path in sorted(lite_dir.glob("*.json")):
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        validation_decision = data.get("validation_decision")
        if not include_valid and validation_decision != "Invalid":
            continue

        scenario_id = data.get("scenario_id") or trace_path.stem
        records.append(
            TraceRecord(
                trace_path=trace_path,
                scenario_id=scenario_id,
                run_number=parse_run_number(trace_path),
                validation_decision=validation_decision,
                validation_rationale=data.get("validation_rationale"),
                interaction_histories=data.get("per_agent_interaction_histories", {}),
                llm_usage_stats=data.get("per_agent_llm_usage_stats", {}),
            )
        )

    return records


def prepare_trace_payload(record: TraceRecord, max_chars: int) -> dict[str, Any]:
    agent_payloads = []
    for agent_index, (agent_id, history) in enumerate(record.interaction_histories.items()):
        usage = record.llm_usage_stats.get(agent_id, {})
        rendered_history = render_history(history, max_chars=max_chars)
        last_action = None
        for message in reversed(history):
            last_action = extract_action_name(message.get("content"))
            if last_action is not None:
                break

        agent_payloads.append(
            {
                "label": normalize_agent_label(agent_index),
                "agent_id": agent_id,
                "message_count": len(history),
                "total_llm_calls": usage.get("total_llm_calls", 0),
                "tool_call_counts": summarize_tool_counts(history),
                "last_action": last_action,
                "history": rendered_history,
            }
        )

    return {
        "scenario_id": record.scenario_id,
        "run_number": record.run_number,
        "trace_path": str(record.trace_path),
        "validation_decision": record.validation_decision,
        "validation_rationale": record.validation_rationale,
        "agent_count": len(agent_payloads),
        "agents": agent_payloads,
    }


def get_engine(model: str, provider: str, endpoint: str) -> LLMEngine:
    builder = LLMEngineBuilder()
    return builder.create_engine(
        LLMEngineConfig(
            model_name=model,
            provider=provider,
            endpoint=endpoint,
        )
    )


def extract_json_object(raw_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate = raw_text.strip()

    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.replace("json\n", "", 1).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM response did not contain a valid JSON object.")


def analyze_single_trace(engine: LLMEngine, prepared_trace: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": TRACE_ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze this failed trace packet and return JSON only.\n\n"
                + json.dumps(prepared_trace, ensure_ascii=False, indent=2)
            ),
        },
    ]
    response_text, metadata = engine(messages)
    analysis = extract_json_object(response_text)
    analysis["llm_metadata"] = metadata or {}
    return analysis


def analyze_single_trace_with_config(
    prepared_trace: dict[str, Any],
    model: str,
    provider: str,
    endpoint: str,
) -> dict[str, Any]:
    engine = get_engine(model, provider, endpoint)
    return analyze_single_trace(engine, prepared_trace)


def aggregate_analyses(engine: LLMEngine, analyses: list[dict[str, Any]]) -> dict[str, Any]:
    payload = []
    for item in analyses:
        payload.append(
            {
                "scenario_id": item["scenario_id"],
                "run_number": item["run_number"],
                "root_cause_summary": item["analysis"].get("root_cause_summary"),
                "failure_modes": item["analysis"].get("failure_modes", []),
                "what_agent_missed": item["analysis"].get("what_agent_missed", []),
                "what_agent_got_wrong": item["analysis"].get("what_agent_got_wrong", []),
                "trajectory_based_explanation": item["analysis"].get(
                    "trajectory_based_explanation"
                ),
            }
        )

    messages = [
        {"role": "system", "content": AGGREGATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Aggregate these per-trace failure analyses into a dataset-level summary. "
                "Return JSON only.\n\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        },
    ]
    response_text, metadata = engine(messages)
    analysis = extract_json_object(response_text)
    analysis["llm_metadata"] = metadata or {}
    return analysis


def render_summary_markdown(
    aggregate_summary: dict[str, Any],
    analyses: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Failure Analysis Summary")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(aggregate_summary.get("dataset_summary", ""))
    lines.append("")

    top_failure_modes = aggregate_summary.get("top_failure_modes", [])
    if top_failure_modes:
        lines.append("## Top Failure Modes")
        lines.append("")
        for item in top_failure_modes:
            mode = item.get("mode", "unknown")
            count = item.get("count", 0)
            summary = item.get("summary", "")
            lines.append(f"- `{mode}` ({count}): {summary}")
        lines.append("")

    scenario_patterns = aggregate_summary.get("scenario_patterns", [])
    if scenario_patterns:
        lines.append("## Scenario Patterns")
        lines.append("")
        for item in scenario_patterns:
            scenario_id = item.get("scenario_id", "unknown")
            runs = ", ".join(str(run) for run in item.get("runs", []))
            common_failure = item.get("common_failure", "")
            lines.append(f"- `{scenario_id}` runs [{runs}]: {common_failure}")
        lines.append("")

    recommendations = aggregate_summary.get("recommended_agent_improvements", [])
    if recommendations:
        lines.append("## Recommended Agent Improvements")
        lines.append("")
        for recommendation in recommendations:
            lines.append(f"- {recommendation}")
        lines.append("")

    lines.append("## Per-Trace Analyses")
    lines.append("")
    for item in analyses:
        analysis = item["analysis"]
        lines.append(
            f"### {item['scenario_id']} run {item['run_number'] if item['run_number'] is not None else '?'}"
        )
        lines.append("")
        lines.append(f"- Root cause: {analysis.get('root_cause_summary', '')}")
        failure_modes = ", ".join(analysis.get("failure_modes", []))
        lines.append(f"- Failure modes: {failure_modes}")
        missed = analysis.get("what_agent_missed", [])
        if missed:
            lines.append(f"- Missed: {'; '.join(missed)}")
        got_wrong = analysis.get("what_agent_got_wrong", [])
        if got_wrong:
            lines.append(f"- Got wrong: {'; '.join(got_wrong)}")
        lines.append(
            f"- Explanation: {analysis.get('trajectory_based_explanation', '')}"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def load_existing_analyses(path: Path) -> dict[tuple[str, int | None], dict[str, Any]]:
    if not path.exists():
        return {}

    cached: dict[tuple[str, int | None], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cached[(item["scenario_id"], item.get("run_number"))] = item
    return cached


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_analysis_item(
    prepared_trace: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": prepared_trace["scenario_id"],
        "run_number": prepared_trace["run_number"],
        "trace_path": prepared_trace["trace_path"],
        "validation_rationale": prepared_trace["validation_rationale"],
        "analysis": analysis,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.api_version:
        os.environ["AZURE_API_VERSION"] = args.api_version

    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (results_dir / DEFAULT_OUTPUT_DIR_NAME).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = output_dir / "prepared_traces.jsonl"
    per_trace_path = output_dir / "per_trace_analysis.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    records = load_trace_records(results_dir, include_valid=args.include_valid)
    if args.max_traces is not None:
        records = records[: args.max_traces]

    if not records:
        raise RuntimeError("No traces matched the requested filters.")

    prepared_traces = [prepare_trace_payload(record, args.max_text_chars) for record in records]
    write_jsonl(prepared_path, prepared_traces)
    logger.info("Prepared %s trace payloads at %s", len(prepared_traces), prepared_path)

    if args.dry_run:
        logger.info("Dry run requested, skipping LLM analysis.")
        return

    existing = {} if args.overwrite else load_existing_analyses(per_trace_path)
    all_analyses: list[dict[str, Any]] = []
    traces_to_analyze: list[dict[str, Any]] = []
    for prepared_trace in prepared_traces:
        cache_key = (prepared_trace["scenario_id"], prepared_trace["run_number"])
        if cache_key in existing:
            logger.info(
                "Using cached analysis for %s run %s",
                prepared_trace["scenario_id"],
                prepared_trace["run_number"],
            )
            all_analyses.append(existing[cache_key])
            continue
        traces_to_analyze.append(prepared_trace)

    if traces_to_analyze:
        logger.info(
            "Submitting %s trace analyses with max concurrency %s",
            len(traces_to_analyze),
            args.max_concurrent_requests,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.max_concurrent_requests)
        ) as executor:
            future_to_trace = {
                executor.submit(
                    analyze_single_trace_with_config,
                    prepared_trace,
                    args.model,
                    args.provider,
                    args.endpoint,
                ): prepared_trace
                for prepared_trace in traces_to_analyze
            }
            for future in concurrent.futures.as_completed(future_to_trace):
                prepared_trace = future_to_trace[future]
                logger.info(
                    "Analyzing %s run %s completed",
                    prepared_trace["scenario_id"],
                    prepared_trace["run_number"],
                )
                analysis = future.result()
                all_analyses.append(build_analysis_item(prepared_trace, analysis))
                write_jsonl(per_trace_path, all_analyses)

    if not args.overwrite and existing:
        seen = {(item["scenario_id"], item["run_number"]) for item in all_analyses}
        for key, item in existing.items():
            if key not in seen:
                all_analyses.append(item)

    all_analyses.sort(key=lambda item: (item["scenario_id"], item["run_number"] or -1))
    write_jsonl(per_trace_path, all_analyses)
    logger.info("Wrote per-trace analyses to %s", per_trace_path)

    engine = get_engine(args.model, args.provider, args.endpoint)
    aggregate_summary = aggregate_analyses(engine, all_analyses)
    summary_json_path.write_text(
        json.dumps(aggregate_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(
        render_summary_markdown(aggregate_summary, all_analyses),
        encoding="utf-8",
    )
    logger.info("Wrote aggregate summary to %s and %s", summary_json_path, summary_md_path)


if __name__ == "__main__":
    main()


#   python scripts/analyze_failure_traces.py --results-dir results/260403-validation-gpt-54-azure --max-text-chars 18000 --max-traces 3
