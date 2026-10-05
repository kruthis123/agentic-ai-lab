from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

import time
import os

load_dotenv()

@dataclass
class ModelResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def call_model(prompt: str) -> ModelResult:
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    return ModelResult(
        text=response.choices[0].message.content,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency_ms=latency_ms
    )
