import os
import re
from enum import Enum
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
    You are a data extraction assistant. Given the following schema in YAML format, extract the corresponding fields from the provided markdown text. Return the extracted data as a JSON object that adheres to the schema. If a field is not present in the text, return it as null. Do not include any additional information or commentary.
"""


class LLMProvider(Enum):
    OLLAMA = "ollama"
    CLOUD = "groq"


class LLMClient:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.client, self.model_name = self._configure_client(self.provider)

    def _configure_client(self, provider: LLMProvider) -> tuple[AsyncOpenAI, str]:
        if provider == LLMProvider.OLLAMA:
            client = AsyncOpenAI(
                base_url="http://localhost:11434/v1", api_key="ollama_dummy_key"
            )
            return client, "llama3.1:8b"

        raise ValueError(f"Unsupported LLM provider: {provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ValueError),
        reraise=True,
    )
    async def request(
        self, markdown_text: str, schema_yaml: str, schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Extract structured data from markdown text using the specified LLM provider and schema.

        Args:
            markdown_text (str): The text extracted from the document in Markdown format.
            schema_yaml (str): The YAML representation of the schema to which the extracted data should conform
            schema (Type[BaseModel]): The Pydantic model class representing the schema.

        Returns:
            BaseModel: An instance of the Pydantic model populated with the extracted data.
        """
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

        raw_content = response.choices[0].message.content
        if not raw_content:
            raise ValueError(
                "No content returned from LLM. Please check the input and schema."
            )

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
        clean_json_str = match.group(1) if match else raw_content.strip()

        try:
            model = schema.model_validate_json(clean_json_str)
        except Exception as e:
            raise ValueError(
                f"Pydantic validation failed: {e}\nPayload: {clean_json_str}"
            )

        if all(value is None for value in model.model_dump().values()):
            raise ValueError(f"All fields are null. Payload: {clean_json_str}")

        return model
