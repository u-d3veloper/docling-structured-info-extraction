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

### 1.3.1 Quick test

1. Open <http://localhost:8000/docs> in a browser.
2. Expand **POST /extract** and click **Try it out**.
3. In the `files` field, click **Add string item**, then choose a PDF or an image
   of a scanned document.
4. In the `schema_file` field, choose the sample schema `schemas/base.yaml`. which corresponds to the sample document given in the requirements.
5. Click **Execute**.

The **Response body** panel shows the YAML report and the **Code** field shows the
HTTP status. A successful run looks like this:

```yaml
- source_file: Copy_BL_225728047-3.pdf
  status: success
  extracted_data:
    bl_number: '225728047'
    vessel_name: CAP SAN ARTEMISSIO
    port_of_loading: SANTOS/BRASIL (BRSSZ)
    port_of_discharge: ANTWERPEN
    place_of_acceptance: null
    ...
```

Fields genuinely absent from the document come back as `null` rather than being
invented — see test 4 in [1.3.3](#133-looking-at-my-validation-tests-results).

### 1.3.2 Testing on your own data and schemas

**Documents.** Accepted extensions are `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`,
`.tif`, `.bmp`, `.webp`, anything else is rejected per file, with the rest of the batch still processed. The service is sized for short scanned documents (1–2 pages); longer ones will exceed the model's context window and degrade silently in quality.

You can select several files at once in the `files` field — they are processed concurrently and each gets its own entry in the report.

**Schemas.** A schema is a flat YAML mapping of `field_name` to a specification:

```yaml
bl_number:
  type: string
  description: "The unique Bill of Lading (B/L) tracking number or document reference."

container_numbers:
  type: list
  description: "A list of container identification numbers associated with the shipment. Must be 4 letters followed by 7 digits (like MRKU7061784). Ensure each item is extracted separately."

```

Supported types: `string`, `integer`, `float`, `boolean`, `list`. Field names must
be valid Python identifiers. Nested objects and typed list items are out of scope
(see section 2).

Every field is optional and defaults to `null`, so the schema describes what to
*look for*, not what must be present. The `description` matters more than it looks:
it is injected verbatim into the model's instructions and is the main lever you
have on extraction quality. Vague descriptions produce vague results; naming the
expected format ("extract the exact date string", "return each item separately")
measurably helps an 8B model.

An invalid schema is rejected with **HTTP 400** before any document is processed, so
you never have to wait for the model to fail.

### 1.3.3 Looking at my validation-tests results

The service was validated against the two real documents provided in the requirements. I adapted the given schema to add description for each field so that the model has a better chance of understanding what is expected. The following table summarises the tests and their results. You can reproduce them by using the `schemas/base.yaml` schema.

#### Test : Basic extraction
- Document: `Copy_BL_225728047.pdf`
- Schema: `schemas/base.yaml`
- Result:
```yaml
- source_file: Copy_BL_225728047.pdf
  status: success
  extracted_data:
    company_name: Maersk A/S
    bl_number: '225728047'
    shipper: ESTRELA COMERCIO E EXPORTADORA DE CAFE LTDA.
    port_of_loading: SANTOS/BRASIL (BRSSZ)
    port_of_discharge: ANTWERPEN
    place_of_acceptance: null
    place_of_delivery: null
    shipped_on_board_date: '2023-03-19'
    vessel_name: CAP SAN ARTEMISSIO
    voyage_number: 310N
    mark: MRKU7061784, MSKU5717181
    container_numbers:
    - MRKU7061784
    - MSKU5717181
    lots: null
```

#### Test : parallel extractions
- Documents: `BL_239386606.pdf`, `Copy_BL_225728047.pdf`
- Schema: `schemas/base.yaml`
- Result:
```yaml
- source_file: Copy_BL_225728047.pdf
  status: success
  extracted_data:
    company_name: Maersk A/S
    bl_number: '225728047'
    shipper: ESTRELA COMERCIO E EXPORTADORA DE CAFE LTDA.
    port_of_loading: SANTOS/BRASIL (BRSSZ)
    port_of_discharge: ANTWERPEN
    place_of_acceptance: null
    place_of_delivery: null
    shipped_on_board_date: '2023-03-19'
    vessel_name: CAP SAN ARTEMISSIO
    voyage_number: 310N
    mark: MRKU7061784, MSKU5717181
    container_numbers:
    - MRKU7061784
    - MSKU5717181
    lots: null
- source_file: BL_239386606.pdf
  status: success
  extracted_data:
    company_name: MAERSK
    bl_number: '239386606'
    shipper: TWENDESHAMBATANZANIA LIMITED P.O.BOX 140MBINGA
    port_of_loading: Dares Salaam
    port_of_discharge: Aqaba
    place_of_acceptance: null
    place_of_delivery: null
    shipped_on_board_date: null
    vessel_name: MERATUS JAYAWIIAYA
    voyage_number: 420N
    mark: MRKU7976385
    container_numbers:
    - MSKU5422072
    - ML-TZ0086988
    - MRKU7976385
    lots: null
```

# 2. Technical choices, architecture and trade-offs

## 2.1 Privacy considerations

Knowing that the service is designed to run locally, no document or schema is sent to any external service. 

This is designed to ensure that sensitive information contained in the documents is not exposed to third parties, and that the user retains full control over their data. 

For such services with ability to be run on small models instead of frontier LLMs, I often recommend and tend to use local models for privacy reasons, as they can be run on the user's own hardware without the need for an internet connection or cloud-based processing.

## 2.2 Performance considerations

With such a service, performance is a key consideration. The extraction service is needed to be fast and efficient at scale with large volumes of documents. 

For such an exercice, I have chosen to use a small model (8B) in a quantized version (4-bit) to reduce memory footprint and improve inference speed as it shows generally good performance for the task at hand.

Regarding the document conversion pipeline, I have chosen to use OCR with layout analysis. This allows for accurate extraction of information from a variety of document formats, including scanned images and PDFs. The conversion pipeline is run with docling and easyOCR as the backend engine. 

There is a trade-off here between the accuracy, the speed and the privacy considerations. The choice of using a small model and running the conversion pipeline locally allows for a good balance between these factors with a pipeline time extraction around 20 seconds on my machine while keeping relative accuracy on structured information with only one field (MARK) not retrievedfrom documents. Still, there is room for improvement in terms of speed and accuracy with a larger model like 14B or 27B for extremely large scale deployements, but that would require more resources (server-like infrastructure). There is no need for frontier models like GPT-4 or Claude, with more than 100B parameters, as they are overkill for this task and would require more resources and infrastructure to run.

### 2.2.1 Note on CPU / GPU execution
 
**The document conversion pipeline is deliberately pinned to CPU.** In
`extractor.py`:
 
```python
accelerator_options = AcceleratorOptions(
    num_threads=cpu_cores,
    device=AcceleratorDevice.CPU,
)
```
 
This is a portability choice, and it is also the safest one when an LLM is running
on the same machine. Two situations are worth distinguishing:
 
**Unified-memory machines (Apple Silicon, AMD, Intel).** VRAM and system RAM share the same physical memory, so this is the reference setup and it works out of the box.
 
**Split-memory machines (Linux or Windows with a separate NVIDIA GPU).** VRAM and
system RAM are separate.
 
Setting `device=AcceleratorDevice.CPU` is normally enough to avoid errors on such devices. If PyTorch
still initialises a CUDA context, `extractor.py` ships with a commented-out hard override at the top of the file:
 
```python
# Uncomment on a machine with a separate GPU (Linux / Windows + NVIDIA) if the
# conversion pipeline competes with Ollama for VRAM. Must stay above the docling
# imports: CUDA_VISIBLE_DEVICES is read when torch is first imported.

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```
 
Uncommenting it makes every CUDA device invisible to this process, so PyTorch falls back to CPU unconditionally. 

If instead you *want* to accelerate conversion on a machine with spare VRAM, switch
`device` to `AcceleratorDevice.CUDA` and expect to trade throughput against thememory Ollama needs. This was not validated here and is out of scope for the
memory Ollama needs. 

## Accuracy considerations

The extraction accuracy is highly dependent on the quality of the LLM but also on the architecture choices. For such an exercice, I still have chosen to implement an architecture with classical patterns and practices to ensure a better quality of the extraction.

### About the schema 

Metadata validation is a key part of the service. The schema is validated before any document is processed. The encoding and the parsing also. 

The injection of user-provided introduce security risks, so the service is designed to validate the schema before any processing. Each field is mapped to a type and the description is injected into the prompt to help the model understand what is expected.

### About the LLM 
We fix the temperature to 0.0 to ensure determinism of the results. The LLM answers is parsed with REGEX to extract the JSON even if the model surrounds it with markdown tags. The extracted JSON is validated with Pydantic to ensure that the types are correct, otherwise an error is raised.

To prevent too many fails, I implemented a retry mechanism with a maximum of 3 retries. The retry mechanism is designed to handle cases where the model fails to produce a valid answer due to its stochastic nature. The exponential backoff is used to avoid overwhelming the model with too many requests in a short period of time. 
