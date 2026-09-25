# SIE (Structured Information Extraction) Service

## About this README

In this README you'll find two main sections: the first section help you launch the service and use it. The second section is a more technical description of the service, its architecture and how it works, along with technical choices as well as trade-offs regarding the exercice requirements.

## About the SIE Service

The goal of the SIE service as described in the requirements is to extract structured information from scanned documents and PDF following a user-given schema. It is designed to be able to process documents in a variety of formats, including images and PDFs, and extract relevant information from them. 

# 1. Launching the service

## 1.1 Requirements

### 1.1.1 Hardware and OS

The service was developed and validated on **macOS 14+ / Apple Silicon (M2)**. It is
platform-independent by design: the document conversion pipeline is pinned to CPU
execution (see [1.2.4](#124-note-on-cpu--gpu-execution)), so nothing in the code
depends on a particular accelerator.

| Platform                        | Tested           | Notes                                                                                                     |
| ------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------- |
| macOS, Apple Silicon (M1/M2/M3) | Yes        | Reference platform. Unified memory, no GPU/CPU split to worry about.                                      |
| Linux, x86_64                   | Yes | See [2.2.1](#221-note-on-cpu--gpu-execution) if you have a discrete NVIDIA GPU.                           |


Minimum practical resources: 4 CPU cores and 16 GB of RAM. The LLM alone
(`llama3.1:8b`, 4-bit quantised) needs roughly 5 GB, and the document conversion
models another 1–2 GB.

### 1.1.2 Python

**Python 3.10 or later** is required.

```bash
python --version   # should print 3.10.x or higher
```

### 1.1.3 Ollama

The service calls a **locally running Ollama instance** through an
OpenAI-compatible endpoint (`http://localhost:11434/v1`). 

Install Ollama from <https://ollama.com/download> (native installers exist for
macOS, Windows and Linux), then pull the model:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```


### 1.1.4 UV
Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`, committed alongside `pyproject.toml`.
 
Install uv (one-line installers, same command family on every platform):
 
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
 
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


## 1.2 Launching the service

### 1.2.1 Dependency installation

```bash
uv sync
```
This creates .venv/ and installs the exact locked versions — no manual venv creation or activation needed.

### 1.2.2 First run: model download

> As the architecture is designed to be local, the first document you submit triggers a one-time download of the document conversion models (layout analysis, table structure, OCR weights) — You can expect a few minutes for this.

### 1.2.3 Starting the two processes

Ollama first — on macOS and Windows the desktop app starts the server
automatically; on Linux, or if you installed the CLI only:

```bash
ollama run ollama run llama3.1:8b-instruct-q4_K_M
```
once the invite loads on your terminal screen you can type /bye. The model stays resident in memory and is still serving requests.

Then launch the API, from the project root:

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

## 1.3 Usage

The service exposes a single working endpoint, `POST /extract`, plus a `GET /health`
probe. It takes one or more documents with one or more formats (PDF, JPG, PNG, Other images formats, ...) and one YAML schema file, and returns a YAML
report.

The easiest way to exercise it is the auto-generated Swagger UI:

```
http://localhost:8000/docs
```
