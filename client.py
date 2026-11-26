import os
import re
from typing import Type
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

SYSTEM_PROMPT = """
You are a data extraction assistant for maritime Bills of Lading. Given the
schema in YAML below, extract the fields from the markdown text and return a
single JSON object matching the schema. Return null for any field not present.
Do not include commentary.

Rules:
1. Values are the content of the field only. Do NOT include section labels
   ("SHIPPING MARK", "Particulars Furnished by Shipper", etc.) in the value.
2. `shipper` is a single entity: its full block (name AND address if present),
   never a list and never concatenated with the consignee.
3. `lots` contains the goods description only, with no weights. Strip any
   "Gross wt. ... KG" suffix that sits on the same row.
4. `port_of_loading` / `port_of_discharge` are the ports WITH their UN/LOCODE
   when present (e.g. "Singapore (SGSIN)"). `place_of_acceptance` /
   `place_of_delivery` are the inland or terminal location (e.g. "Singapore
   CY") — they are distinct fields; do not swap them.
5. When a table shifts a column off by one, anchor each value to its header
   label, not to its visual position.
6. If a value is ambiguous or not clearly readable, return null rather than
   guessing or copying a nearby cell.

schema:
{schema_yaml}
"""


class LLMClient:
    """Async client for the local Ollama server over its OpenAI-compatible API.

    The model tag is read from ``OLLAMA_MODEL`` (default ``llama3.1:latest``).
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="http://localhost:11434/v1", api_key="ollama_dummy_key"
        )
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ValueError),
        reraise=True,
    )
    async def request(
        self, markdown_text: str, schema_yaml: str, schema: Type[BaseModel]
    ) -> BaseModel:
        """Call the LLM, parse the JSON response, validate against ``schema``."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(schema_yaml=schema_yaml),
                },
                {"role": "user", "content": markdown_text},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or ""
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
        clean_json_str = match.group(1) if match else raw_content.strip()

        try:
            model = schema.model_validate_json(clean_json_str)
        except Exception as e:
            raise ValueError(
                f"Pydantic validation failed: {e}\nPayload: {clean_json_str}"
            )

        # All-null response usually means the prompt wasn't understood; retry via tenacity.
        if all(value is None for value in model.model_dump().values()):
            raise ValueError(f"All fields are null. Payload: {clean_json_str}")

        return model
