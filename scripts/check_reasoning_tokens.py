#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from litellm import completion
from openai import AzureOpenAI, OpenAI

AZURE_OPENAI_DEFAULT_SCOPE = "https://cognitiveservices.azure.com/.default"
AZURE_TRAPI_SCOPE = "api://trapi/.default"
DEFAULT_AZURE_ENDPOINT = "https://societalllm.openai.azure.com/"
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_EFFORTS = "default,none,minimal,low,medium,high,xhigh"
DEFAULT_TRANSPORTS = "sdk-chat,sdk-responses,litellm-chat"
DEFAULT_PROMPT = (
    "Solve this exactly: How many 5-letter strings over {A,B,C} contain exactly "
    "two As and no identical adjacent letters? Reply with the final integer and "
    "one short justification."
)
MISSING = object()


@dataclass
class ProbeResult:
    transport: str
    effort_label: str
    requested_effort: str | None
    success: bool
    usage: dict[str, Any] | None = None
    preview: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def reasoning_observation(self) -> tuple[str | None, bool, Any]:
        usage = self.usage or {}
        for path in (
            "completion_tokens_details.reasoning_tokens",
            "output_tokens_details.reasoning_tokens",
            "reasoning_tokens",
        ):
            value = get_path(usage, path)
            if value is not MISSING:
                return path, True, value
        return None, False, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether gpt-5.4 reasoning token usage is missing because of "
            "reasoning_effort, because the interface does not return the field, "
            "or because LiteLLM drops it."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        default="azure",
        choices=["azure", "openai"],
        help="Provider used for both the direct SDK probe and the LiteLLM probe.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_API_BASE") or DEFAULT_AZURE_ENDPOINT,
        help="Azure endpoint or custom OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_API_VERSION") or DEFAULT_AZURE_API_VERSION,
        help="Azure API version used by the direct SDK and LiteLLM Azure probes.",
    )
    parser.add_argument(
        "--efforts",
        default=DEFAULT_EFFORTS,
        help=(
            "Comma-separated reasoning efforts to test. Use `default` to omit the "
            "parameter entirely."
        ),
    )
    parser.add_argument(
        "--transports",
        default=DEFAULT_TRANSPORTS,
        help=(
            "Comma-separated transports to run. Supported values: sdk-chat, "
            "sdk-responses, litellm-chat."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for each probe call.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write the full probe results as JSON.",
    )
    return parser.parse_args()


def split_csv(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",")]
    return [item for item in values if item]


def normalize_effort(raw_effort: str) -> tuple[str, str | None]:
    label = raw_effort.strip()
    if label.lower() in {"default", "unset", "omit"}:
        return "default", None
    return label, label


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(inner) for inner in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="json", exclude_none=False))
        except TypeError:
            return to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): to_jsonable(inner)
            for key, inner in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def get_path(data: Any, dotted_path: str) -> Any:
    current = data
    for segment in dotted_path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        return MISSING
    return current


def get_text_preview(text: str | None, limit: int = 120) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_azure_ad_token_provider(endpoint: str | None):
    try:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
    except ImportError as exc:
        raise RuntimeError(
            "azure-identity is required for Azure AD authentication."
        ) from exc

    if endpoint and "trapi.research.microsoft.com" in endpoint:
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        scope = AZURE_TRAPI_SCOPE
    else:
        credential = AzureCliCredential()
        scope = AZURE_OPENAI_DEFAULT_SCOPE

    return get_bearer_token_provider(credential, scope)


def build_sdk_client(args: argparse.Namespace) -> AzureOpenAI | OpenAI:
    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for --provider openai.")
        base_url = args.endpoint if args.endpoint and args.endpoint != DEFAULT_AZURE_ENDPOINT else None
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    if api_key:
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=args.endpoint,
            api_version=args.api_version,
        )

    azure_ad_token = os.getenv("AZURE_AD_TOKEN")
    if azure_ad_token:
        return AzureOpenAI(
            azure_ad_token=azure_ad_token,
            azure_endpoint=args.endpoint,
            api_version=args.api_version,
        )

    return AzureOpenAI(
        azure_endpoint=args.endpoint,
        azure_ad_token_provider=build_azure_ad_token_provider(args.endpoint),
        api_version=args.api_version,
    )


def build_litellm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
    }

    if args.provider == "openai":
        kwargs["custom_llm_provider"] = "openai"
        if args.endpoint and args.endpoint != DEFAULT_AZURE_ENDPOINT:
            kwargs["api_base"] = args.endpoint
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for --provider openai.")
        kwargs["api_key"] = api_key
        return kwargs

    kwargs["custom_llm_provider"] = "azure"
    kwargs["api_base"] = args.endpoint
    kwargs["api_version"] = args.api_version

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
        return kwargs

    azure_ad_token = os.getenv("AZURE_AD_TOKEN")
    if azure_ad_token:
        kwargs["azure_ad_token"] = azure_ad_token
        return kwargs

    kwargs["azure_ad_token_provider"] = build_azure_ad_token_provider(args.endpoint)
    return kwargs


def probe_sdk_chat(
    client: AzureOpenAI | OpenAI,
    args: argparse.Namespace,
    effort_label: str,
    requested_effort: str | None,
) -> ProbeResult:
    request_kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    if requested_effort is not None:
        request_kwargs["reasoning_effort"] = requested_effort

    try:
        response = client.chat.completions.create(**request_kwargs)
        preview = response.choices[0].message.content
        return ProbeResult(
            transport="sdk-chat",
            effort_label=effort_label,
            requested_effort=requested_effort,
            success=True,
            usage=to_jsonable(getattr(response, "usage", None)),
            preview=get_text_preview(preview),
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            transport="sdk-chat",
            effort_label=effort_label,
            requested_effort=requested_effort,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def probe_sdk_responses(
    client: AzureOpenAI | OpenAI,
    args: argparse.Namespace,
    effort_label: str,
    requested_effort: str | None,
) -> ProbeResult:
    request_kwargs: dict[str, Any] = {
        "model": args.model,
        "input": args.prompt,
    }
    if requested_effort is not None:
        request_kwargs["reasoning"] = {"effort": requested_effort}

    try:
        response = client.responses.create(**request_kwargs)
        preview = getattr(response, "output_text", None)
        return ProbeResult(
            transport="sdk-responses",
            effort_label=effort_label,
            requested_effort=requested_effort,
            success=True,
            usage=to_jsonable(getattr(response, "usage", None)),
            preview=get_text_preview(preview),
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            transport="sdk-responses",
            effort_label=effort_label,
            requested_effort=requested_effort,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def probe_litellm_chat(
    args: argparse.Namespace,
    effort_label: str,
    requested_effort: str | None,
) -> ProbeResult:
    request_kwargs = build_litellm_kwargs(args)
    if requested_effort is not None:
        request_kwargs["reasoning_effort"] = requested_effort

    try:
        response = completion(**request_kwargs)
        preview = response.choices[0].message.content
        usage = response.get("usage")
        return ProbeResult(
            transport="litellm-chat",
            effort_label=effort_label,
            requested_effort=requested_effort,
            success=True,
            usage=to_jsonable(usage),
            preview=get_text_preview(preview),
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            transport="litellm-chat",
            effort_label=effort_label,
            requested_effort=requested_effort,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def infer_conclusions(results: list[ProbeResult]) -> list[str]:
    conclusions: list[str] = []

    by_transport: dict[str, list[ProbeResult]] = {}
    for result in results:
        by_transport.setdefault(result.transport, []).append(result)

    sdk_chat = [result for result in by_transport.get("sdk-chat", []) if result.success]
    sdk_responses = [
        result for result in by_transport.get("sdk-responses", []) if result.success
    ]
    litellm_chat = [
        result for result in by_transport.get("litellm-chat", []) if result.success
    ]

    if sdk_chat:
        present = [result for result in sdk_chat if result.reasoning_observation()[1]]
        nonzero = [
            result
            for result in present
            if isinstance(result.reasoning_observation()[2], (int, float))
            and result.reasoning_observation()[2] > 0
        ]
        if not present:
            conclusions.append(
                "Direct chat.completions calls did not return any reasoning-token field on this endpoint."
            )
        elif not nonzero:
            conclusions.append(
                "Direct chat.completions returned a reasoning-token field, but every successful probe reported 0."
            )
        else:
            low_like = {
                result.effort_label
                for result in present
                if result.effort_label in {"default", "none", "minimal", "low"}
                and (result.reasoning_observation()[2] or 0) == 0
            }
            high_like = {
                result.effort_label
                for result in nonzero
                if result.effort_label in {"medium", "high", "xhigh"}
            }
            if low_like and high_like:
                conclusions.append(
                    "Direct chat.completions only surfaced non-zero reasoning tokens at higher reasoning_effort values."
                )

    minimal_errors = [
        result
        for result in results
        if result.effort_label == "minimal" and not result.success
    ]
    if minimal_errors:
        conclusions.append(
            "`minimal` failed in at least one transport, so it may not be supported by this model/provider combination."
        )

    if sdk_chat and sdk_responses:
        sdk_chat_has_field = any(result.reasoning_observation()[1] for result in sdk_chat)
        sdk_responses_has_field = any(
            result.reasoning_observation()[1] for result in sdk_responses
        )
        if not sdk_chat_has_field and sdk_responses_has_field:
            conclusions.append(
                "The Responses API exposed reasoning-token detail while chat.completions did not, so the missing field is interface-specific."
            )

    if sdk_chat and litellm_chat:
        sdk_chat_has_field = any(result.reasoning_observation()[1] for result in sdk_chat)
        litellm_has_field = any(
            result.reasoning_observation()[1] for result in litellm_chat
        )
        if sdk_chat_has_field and not litellm_has_field:
            conclusions.append(
                "LiteLLM dropped or reshaped the reasoning-token field relative to the direct chat.completions response."
            )
        elif sdk_chat_has_field == litellm_has_field:
            conclusions.append(
                "LiteLLM matched the direct chat.completions field presence in this probe."
            )

    if not conclusions:
        conclusions.append(
            "The probe was inconclusive; inspect the per-call usage blobs below."
        )

    return conclusions


def result_to_json(result: ProbeResult) -> dict[str, Any]:
    path, present, value = result.reasoning_observation()
    return {
        "transport": result.transport,
        "effort_label": result.effort_label,
        "requested_effort": result.requested_effort,
        "success": result.success,
        "reasoning_path": path,
        "reasoning_field_present": present,
        "reasoning_value": value,
        "preview": result.preview,
        "usage": result.usage,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def print_header(args: argparse.Namespace, efforts: list[str], transports: list[str]) -> None:
    print("=== Reasoning Token Probe ===")
    print(f"provider:    {args.provider}")
    print(f"model:       {args.model}")
    print(f"endpoint:    {args.endpoint}")
    if args.provider == "azure":
        print(f"api_version: {args.api_version}")
    print(f"efforts:     {', '.join(efforts)}")
    print(f"transports:  {', '.join(transports)}")
    print()


def print_summary(results: list[ProbeResult]) -> None:
    print("=== Summary Table ===")
    print(
        "transport\teffort\tstatus\treasoning_path\treasoning_field_present\treasoning_value"
    )
    for result in results:
        path, present, value = result.reasoning_observation()
        status = "ok" if result.success else f"error:{result.error_type}"
        value_text = json.dumps(value, ensure_ascii=False) if present else "-"
        print(
            f"{result.transport}\t{result.effort_label}\t{status}\t"
            f"{path or '-'}\t{str(present).lower()}\t{value_text}"
        )
    print()

    print("=== Conclusion ===")
    for line in infer_conclusions(results):
        print(f"- {line}")
    print()


def print_details(results: list[ProbeResult]) -> None:
    print("=== Per-Call Details ===")
    for result in results:
        print(f"[{result.transport}] effort={result.effort_label}")
        if not result.success:
            print(f"  status: error")
            print(f"  error_type: {result.error_type}")
            print(f"  error_message: {result.error_message}")
            print()
            continue

        path, present, value = result.reasoning_observation()
        print("  status: ok")
        print(f"  preview: {result.preview}")
        print(f"  reasoning_path: {path}")
        print(f"  reasoning_field_present: {str(present).lower()}")
        print(f"  reasoning_value: {json.dumps(value, ensure_ascii=False)}")
        print("  usage:")
        print(json.dumps(result.usage, ensure_ascii=False, indent=2, sort_keys=True))
        print()


def validate_transports(transports: list[str]) -> None:
    supported = {"sdk-chat", "sdk-responses", "litellm-chat"}
    unknown = [transport for transport in transports if transport not in supported]
    if unknown:
        raise SystemExit(f"Unsupported transport(s): {', '.join(unknown)}")


def main() -> int:
    args = parse_args()
    efforts = split_csv(args.efforts)
    transports = split_csv(args.transports)

    if not efforts:
        raise SystemExit("At least one effort must be provided.")
    if not transports:
        raise SystemExit("At least one transport must be provided.")

    validate_transports(transports)
    print_header(args, efforts, transports)

    normalized_efforts = [normalize_effort(effort) for effort in efforts]
    results: list[ProbeResult] = []

    sdk_client: AzureOpenAI | OpenAI | None = None
    if any(transport.startswith("sdk-") for transport in transports):
        sdk_client = build_sdk_client(args)

    for effort_label, requested_effort in normalized_efforts:
        if "sdk-chat" in transports:
            assert sdk_client is not None
            results.append(probe_sdk_chat(sdk_client, args, effort_label, requested_effort))
        if "sdk-responses" in transports:
            assert sdk_client is not None
            results.append(
                probe_sdk_responses(sdk_client, args, effort_label, requested_effort)
            )
        if "litellm-chat" in transports:
            results.append(probe_litellm_chat(args, effort_label, requested_effort))

    print_summary(results)
    print_details(results)

    if args.json_output:
        payload = {
            "provider": args.provider,
            "model": args.model,
            "endpoint": args.endpoint,
            "api_version": args.api_version,
            "results": [result_to_json(result) for result in results],
            "conclusions": infer_conclusions(results),
        }
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    return 0 if any(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
