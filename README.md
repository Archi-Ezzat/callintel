# CallIntel

CallIntel is a local-first pipeline for:

1. Normalizing call audio
2. Splitting it into chunks
3. Transcribing with Whisper
4. Merging transcripts
5. Flagging trigger terms
6. Producing a risk report and summary

## Project Layout

```text
callintel/
  README.md
  .env.example
  requirements.txt

  models/
    whisper-large-v3/
    sentiment/
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
          combined_risk.json
          sentiment.json
          diarization.json
    work/
      0001_callname/
        audio/
        chunks/
        transcripts/
        merged/
        analysis/

  scripts/
    setup.py
    download_models.py

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

Fresh machine, required runtime only:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/setup.py
```

That setup command will:

- create `.env` from `.env.example` if needed
- install the project dependencies
- download the required Whisper model into `models/whisper-large-v3`

Recommended full setup:

```bash
python scripts/setup.py --all
```

That also downloads the sentiment model and installs diarization support.

Manual model download is still available:

```bash
python scripts/download_models.py --whisper
python scripts/download_models.py --all
```

Notes:

- diarization needs `HF_TOKEN` in `.env`
- the local LLM is optional; if `models/llm` is empty, the pipeline falls back to heuristic reporting
- for MP3 and some other codecs, `ffmpeg` may still be required on the machine

To force Arabic transcription and Arabic output:

- `LANGUAGE=ar`
- `LLM_OUTPUT_LANGUAGE=ar`

## Classifier

This project supports a rule-based risk engine and an optional ML classifier.

Seed samples live in:

`data/training/seed.jsonl`

Create your training file by copying:

`data/training/seed.jsonl` -> `data/training/train.jsonl`

You can add more data or generate synthetic samples, then optionally create:

`data/training/val.jsonl`

Train the classifier with:

```bash
.\.venv\Scripts\python.exe -m src.train_classifier
```

This fine-tunes `UBC-NLP/MARBERTv2` and writes to:

`models/classifier`

Then set:

`CLASSIFIER_MODEL_PATH=models/classifier`

## Input

Drop your audio files in:

`data/input_calls`

Supported formats include `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, and `.aac`.

## Run

```bash
python -m src.pipeline
```

Useful options:

```bash
python -m src.pipeline --file data/input_calls/sample_call.mp3
python -m src.pipeline --skip-llm
python -m src.pipeline --skip-sentiment
python -m src.pipeline --skip-diarize
python -m src.pipeline --redact-pii
python -m src.pipeline --force
```

## Output

Final deliverables go to:

`data/output/<audio filename>/`

Inside you will see:

- `Transcript/` with `transcript.txt` and `transcript.json`
- `LLM_Justification/` with the risk, sentiment, diarization, and summary artifacts

Intermediate artifacts still live in:

`data/work/0001_callname`
