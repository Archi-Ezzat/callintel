from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

    if output_language == "ar":
        summary = "تمت معالجة المحادثة محليًا. تم استخدام كثافة الكلمات التحذيرية كإشارة رئيسية للمخاطر."
        key_points = [
            f"طول النص: {total_words} كلمة",
            f"عدد الكلمات التحذيرية: {total_hits}",
            f"أهم الكلمات التحذيرية: {', '.join(top_terms) if top_terms else 'لا يوجد'}",
        ]
    else:
        summary = (
            "Conversation processed locally. Trigger density was used as the main risk signal."
        )
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


def _local_llm_summary(
    text: str,
    trigger_payload: dict[str, Any],
    llm_model_path: Path,
    quantization: str | None = None,
    output_language: str = "en",
) -> dict[str, Any] | None:
    if not llm_model_path.exists():
        return None

    if output_language == "ar":
        prompt = (
            "حلّل نص مكالمة دعم العملاء التالية. "
            "أعد ملخصًا موجزًا للمخاطر، وأبرز القضايا، والتوصيات التالية للمتابعة. "
            "الرجاء كتابة الإجابة بالعربية الفصحى وبشكل مختصر وواضح.\n\n"
            f"النص:\n{text[:12000]}\n\n"
            f"عدد الكلمات التحذيرية: {json.dumps(trigger_payload.get('counts', {}), ensure_ascii=False)}"
        )
    else:
        prompt = (
            "Analyze the following customer support call transcript. "
            "Return a concise risk summary, key issues, and recommended follow-up.\n\n"
            f"Transcript:\n{text[:12000]}\n\n"
            f"Trigger counts: {json.dumps(trigger_payload.get('counts', {}), ensure_ascii=False)}"
        )
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

    lines = []
    if output_language == "ar":
        lines.append("الملخص")
        lines.append(summary if summary else "لم يتم إنشاء ملخص.")
    else:
        lines.append("Summary")
        lines.append(summary if summary else "No summary generated.")
    lines.append("")
    if output_language == "ar":
        lines.append("أهم النقاط")
        if key_points:
            for item in key_points:
                lines.append(f"- {item}")
        else:
            lines.append("- لا يوجد")
    else:
        lines.append("Key Points")
        if key_points:
            for item in key_points:
                lines.append(f"- {item}")
        else:
            lines.append("- None")
    lines.append("")
    if output_language == "ar":
        lines.append("الدرجات")
        if scores:
            for key, value in scores.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- لا يوجد")
    else:
        lines.append("Scores")
        if scores:
            for key, value in scores.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- None")

    return "\n".join(lines).strip() + "\n"


def evaluate_transcript(
    merged_json_path: Path,
    triggers_json_path: Path,
    analysis_dir: Path,
    llm_model_path: Path | None = None,
    llm_quantization: str | None = None,
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
