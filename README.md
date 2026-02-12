# CallIntel

CallIntel is a local-first pipeline for:

1. Normalizing call audio
2. Splitting it into chunks
3. Transcribing with Whisper
4. Merging transcripts
5. Flagging trigger terms
6. Producing an LLM-style analysis report and score

## Project Layout

```text
callintel/
  README.md
  .env.example
  requirements.txt

  models/
    whisper-large-v3/
    llm/

  data/
    input_calls/
    output/
      callname.mp3/
        Transcript/
          transcript.txt
          transcript.json
        LLM_Justification/
          llm_report.txt
          llm_report.json
          score.json
    work/
      0001_callname/
        audio/
        chunks/
        transcripts/
        merged/
        analysis/

  src/
    config.py
    utils_audio.py
    transcribe.py
    merge.py
    triggers.py
    evaluate_llm.py
    pipeline.py
```

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy environment defaults:

```bash
copy .env.example .env
```

## Model Preparation

Place a Whisper model snapshot in:

`models/whisper-large-v3`

Optionally place a local text-generation model in:

`models/llm`

If `models/llm` is empty, the pipeline falls back to deterministic heuristic reporting.

You can enable 4-bit quantization for the local LLM by setting:

`LLM_QUANTIZATION=4bit`

To force Arabic transcription and Arabic LLM reports:

- `LANGUAGE=ar`
- `LLM_OUTPUT_LANGUAGE=ar`

## Classifier (Multi-Path Accuracy)

This project supports a rule-based risk engine **and** an optional ML classifier.

### Seed data

Seed samples live in:

`data/training/seed.jsonl`

Create your training file by copying:

`data/training/seed.jsonl` → `data/training/train.jsonl`

You can add more data or generate synthetic samples, then (optionally) create:

`data/training/val.jsonl`

### Train the classifier

```bash
.\.venv\Scripts\python.exe -m src.train_classifier
```

This fine-tunes `UBC-NLP/MARBERTv2` and writes to:

`models/classifier`

### Use the classifier in the pipeline

Set:

`CLASSIFIER_MODEL_PATH=models/classifier`

## Input

Drop your audio files in:

`data/input_calls`

Supported formats include `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`.

## Run

```bash
python -m src.pipeline
```

Useful options:

```bash
python -m src.pipeline --file data/input_calls/sample_call.mp3
python -m src.pipeline --skip-llm
python -m src.pipeline --force
```

## Output

Final deliverables go to:

`data/output/<audio filename>/`

Inside you will see exactly two folders:

- `Transcript/` (contains `transcript.txt` and `transcript.json`)
- `LLM_Justification/` (contains `llm_report.txt`, `llm_report.json`, `score.json`,
  plus `risk_detail.json` and `risk_summary.txt`)

Intermediate artifacts still live in the workspace directory:

`data/work/0001_callname`

Artifacts:

- `audio/normalized.wav`
- `chunks/chunk_000.wav`, ...
- `transcripts/chunk_000.json`, ...
- `merged/transcript_full.txt`
- `merged/transcript_full.json`
- `analysis/triggers.json`
- `analysis/llm_report.json`
- `analysis/score.json`
