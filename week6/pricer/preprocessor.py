from typing import Literal, cast

from litellm import completion
from litellm.types.utils import ModelResponse

DEFAULT_MODEL_NAME = "groq/openai/gpt-oss-20b"
DEFAULT_REASONING_EFFORT: Literal[
    "none", "minimal", "low", "medium", "high", "default"
] = "low"

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        reasoning_effort: Literal[
            "none", "minimal", "low", "medium", "high", "default"
        ] = DEFAULT_REASONING_EFFORT,
    ):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def messages_for(self, text: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        messages = self.messages_for(text)
        response = cast(
            ModelResponse,
            completion(
                messages=messages,
                model=self.model_name,
                reasoning_effort=cast(
                    Literal["none", "minimal", "low", "medium", "high", "default"],
                    self.reasoning_effort,
                ),
            ),
        )
        usage = response.usage  # type: ignore[attr-defined]
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
        self.total_cost += response._hidden_params["response_cost"]
        content = response.choices[0].message.content  # type: ignore[union-attr]
        return content or ""
