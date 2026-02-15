from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


def _heuristic_summary(
    text: str, trigger_payload: dict[str, Any], output_language: str = "en"
) -> dict[str, Any]:
    words = text.split()
    total_words = len(words)
    total_hits = int(trigger_payload.get("total_hits", 0))

    risk_score = min(100, 15 + total_hits * 10)
    clarity_score = 80 if total_words > 40 else 55
    compliance_score = max(10, 90 - total_hits * 8)

    top_terms = sorted(
        trigger_payload.get("counts", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_terms = [term for term, count in top_terms if count > 0][:5]

    summary = "Conversation processed locally. Trigger density was used as the main risk signal."
    key_points = [
        f"Transcript length: {total_words} words",
        f"Trigger hits: {total_hits}",
        f"Top trigger terms: {', '.join(top_terms) if top_terms else 'none'}",
    ]

    return {
        "mode": "heuristic",
        "summary": summary,
        "key_points": key_points,
        "scores": {
            "risk_score": risk_score,
            "clarity_score": clarity_score,
            "compliance_score": compliance_score,
            "overall_score": round((100 - risk_score + clarity_score + compliance_score) / 3, 2),
        },
    }


def _build_prompt(text: str, trigger_payload: dict[str, Any], output_language: str) -> str:
    if output_language == "ar":
        return (
            "Analyze this customer support call transcript and provide a concise Arabic risk summary, "
            "key issues, and recommended follow-up actions.\\n\\n"
            f"Transcript:\\n{text[:12000]}\\n\\n"
            f"Trigger counts: {json.dumps(trigger_payload.get('counts', {}), ensure_ascii=False)}"
        )
    return (
        "Analyze the following customer support call transcript. "
        "Return a concise risk summary, key issues, and recommended follow-up.\\n\\n"
        f"Transcript:\\n{text[:12000]}\\n\\n"
        f"Trigger counts: {json.dumps(trigger_payload.get('counts', {}), ensure_ascii=False)}"
    )


def _normalize_api_base(api_base_url: str) -> str:
    return api_base_url.rstrip("/")


def _lmstudio_api_get(api_base_url: str, endpoint: str) -> dict[str, Any]:
    req = url_request.Request(
        f"{_normalize_api_base(api_base_url)}/{endpoint.lstrip('/')}",
        method="GET",
    )
    with url_request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _lmstudio_api_post(
    api_base_url: str, endpoint: str, payload: dict[str, Any]
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = url_request.Request(
        f"{_normalize_api_base(api_base_url)}/{endpoint.lstrip('/')}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with url_request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _discover_lmstudio_model(api_base_url: str) -> str | None:
    try:
        payload = _lmstudio_api_get(api_base_url, "models")
    except Exception:
        return None
    models = payload.get("data")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
    return None


def _extract_chat_text(chat_response: dict[str, Any]) -> str:
    choices = chat_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\\n".join(parts).strip()
    return ""


def _lmstudio_llm_summary(
    text: str,
    trigger_payload: dict[str, Any],
    llm_model_path: Path,
    api_base_url: str,
    api_model: str | None,
    output_language: str = "en",
) -> dict[str, Any] | None:
    model_name = api_model or _discover_lmstudio_model(api_base_url) or llm_model_path.stem
    prompt = _build_prompt(text, trigger_payload, output_language=output_language)
    try:
        response = _lmstudio_api_post(
            api_base_url,
            "chat/completions",
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a risk analyst for customer-support calls. "
                            "Focus on safety concerns, intent, and escalation recommendations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 220,
            },
        )
        answer = _extract_chat_text(response)
        if not answer:
            return None
        base = _heuristic_summary(text, trigger_payload, output_language=output_language)
        base["mode"] = "lmstudio_llm"
        base["summary"] = answer[:4000]
        return base
    except (url_error.URLError, TimeoutError, OSError, ValueError, KeyError):
        return None


def _local_llm_summary(
    text: str,
    trigger_payload: dict[str, Any],
    llm_model_path: Path,
    quantization: str | None = None,
    api_base_url: str = "http://127.0.0.1:1234/v1",
    api_model: str | None = None,
    output_language: str = "en",
) -> dict[str, Any] | None:
    if not llm_model_path.exists():
        return None

    if llm_model_path.is_file() and llm_model_path.suffix.lower() == ".gguf":
        return _lmstudio_llm_summary(
            text,
            trigger_payload,
            llm_model_path,
            api_base_url=api_base_url,
            api_model=api_model,
            output_language=output_language,
        )

    if not llm_model_path.is_dir():
        return None

    prompt = _build_prompt(text, trigger_payload, output_language=output_language)
    try:
        generation_config = GenerationConfig(
            max_new_tokens=220,
            do_sample=False,
        )
        if quantization in {"4bit", "4-bit", "int4"}:
            try:
                from transformers import BitsAndBytesConfig
            except Exception:
                return None

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(llm_model_path), local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                local_files_only=True,
                quantization_config=quant_config,
                device_map="auto",
            )
            model.generation_config = generation_config
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
            )
        else:
            device = 0 if torch.cuda.is_available() else -1
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            generator = pipeline(
                "text-generation",
                model=str(llm_model_path),
                local_files_only=True,
                device=device,
                dtype=torch_dtype,
                generation_config=generation_config,
            )
        generated = generator(prompt)[0]["generated_text"]
        answer = generated[len(prompt) :].strip() if generated.startswith(prompt) else generated
        base = _heuristic_summary(text, trigger_payload, output_language=output_language)
        base["mode"] = "local_llm_4bit" if quantization in {"4bit", "4-bit", "int4"} else "local_llm"
        base["summary"] = answer[:4000]
        return base
    except Exception:
        return None


def format_llm_report_text(llm_report: dict[str, Any], output_language: str = "en") -> str:
    summary = (llm_report.get("summary") or "").strip()
    key_points = llm_report.get("key_points") or []
    scores = llm_report.get("scores") or {}

    lines = ["Summary", summary if summary else "No summary generated.", "", "Key Points"]
    if key_points:
        for item in key_points:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Scores")
    if scores:
        for key, value in scores.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- None")

    return "\\n".join(lines).strip() + "\\n"


def evaluate_transcript(
    merged_json_path: Path,
    triggers_json_path: Path,
    analysis_dir: Path,
    llm_model_path: Path | None = None,
    llm_quantization: str | None = None,
    llm_api_base_url: str = "http://127.0.0.1:1234/v1",
    llm_api_model: str | None = None,
    llm_output_language: str = "en",
) -> dict[str, Any]:
    analysis_dir.mkdir(parents=True, exist_ok=True)

    merged_payload = json.loads(merged_json_path.read_text(encoding="utf-8"))
    trigger_payload = json.loads(triggers_json_path.read_text(encoding="utf-8"))
    text = merged_payload.get("text", "")

    llm_report = None
    if llm_model_path is not None:
        llm_report = _local_llm_summary(
            text,
            trigger_payload,
            llm_model_path,
            quantization=llm_quantization,
            api_base_url=llm_api_base_url,
            api_model=llm_api_model,
            output_language=llm_output_language,
        )
    if llm_report is None:
        llm_report = _heuristic_summary(text, trigger_payload, output_language=llm_output_language)

    score_payload = {
        "overall_score": llm_report["scores"]["overall_score"],
        "risk_score": llm_report["scores"]["risk_score"],
        "clarity_score": llm_report["scores"]["clarity_score"],
        "compliance_score": llm_report["scores"]["compliance_score"],
    }

    llm_report_path = analysis_dir / "llm_report.json"
    score_path = analysis_dir / "score.json"

    llm_report_path.write_text(
        json.dumps(llm_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    score_path.write_text(json.dumps(score_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"llm_report": llm_report, "score": score_payload}
