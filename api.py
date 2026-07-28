import asyncio
import yaml
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
from typing import Type
import logging
from extractor import Extractor
from client import LLMClient, LLMProvider
from models import build_dynamic_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}  # Aligned with docling InputFormat.IMAGE
SCHEMA_SUFFIXES = {".yaml", ".yml"}
MAX_CONCURRENT_DOCS = 5
MAX_FILE_BYTES = 10 * 1024 * 1024

app_states = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_states["extractor"] = Extractor()
    app_states["llm_client"] = LLMClient(provider=LLMProvider.OLLAMA)
    app_states["executor"] = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOCS)
    yield
    app_states["executor"].shutdown(wait=True)
    app_states.clear()


app = FastAPI(lifespan=lifespan)


def load_schema(raw: bytes, filename: str) -> tuple[str, Type[BaseModel]]:
    if Path(filename).suffix.lower() not in SCHEMA_SUFFIXES:
        raise ValueError(f"'{filename}' n'est pas un fichier .yaml/.yml.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Le fichier de schéma doit être encodé en UTF-8.")

    schema_dict = yaml.safe_load(text)
    if not schema_dict:
        raise ValueError("Le schéma YAML est vide.")
    return text, build_dynamic_model(schema_dict)


async def process_document(
    file: UploadFile,
    extractor: Extractor,
    llm_client: LLMClient,
    schema_yaml: str,
    schema: Type[BaseModel],
    executor: ThreadPoolExecutor,
) -> dict:
    filename = file.filename or "unknown"
    try:
        if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Format non supporté : '{filename}'.")

        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_BYTES:
            raise ValueError(
                f"'{filename}' dépasse {MAX_FILE_BYTES // (1024 * 1024)} Mo."
            )

        loop = asyncio.get_running_loop()

        markdown_text = await loop.run_in_executor(
            executor, extractor.extract_from_bytes, file_bytes, filename
        )

        structured_data = await llm_client.request(
            markdown_text, schema_yaml=schema_yaml, schema=schema
        )

        return {
            "source_file": filename,
            "status": "success",
            "extracted_data": structured_data.model_dump(),
        }
    except Exception as e:
        return {"source_file": filename, "status": "error", "error": str(e)}


@app.post("/extract")
async def extract_documents(
    files: list[UploadFile] = File(
        ..., description="PDF Document or image files to process"
    ),
    schema_file: UploadFile = File(
        ..., description="YAML file describing the fields to extract"
    ),
):
    try:
        schema_yaml, DynamicModel = load_schema(
            await schema_file.read(), schema_file.filename or ""
        )
    except (yaml.YAMLError, ValueError) as e:
        logger.error(f"Schema error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Schema error, please check the schema file you provided.",
        )

    tasks = [
        process_document(
            file,
            app_states["extractor"],
            app_states["llm_client"],
            schema_yaml,
            DynamicModel,
            app_states["executor"],
        )
        for file in files
    ]
    batch_results = await asyncio.gather(*tasks)

    yaml_report = yaml.dump(
        batch_results, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    all_failed = all(result["status"] == "error" for result in batch_results)
    return Response(
        content=yaml_report,
        media_type="application/x-yaml",
        status_code=400 if all_failed else 200,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
