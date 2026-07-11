"""Single source of truth for the product-rewrite prompt and model, plus a small
interactive Preprocessor. `batchb.py` imports these constants so the prompt /
message-building logic lives in exactly one place (previously duplicated across
day2, batch and preprocessor).
"""

from typing import Literal, cast

from litellm import completion
from litellm.types.utils import ModelResponse

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "default"]

MODEL = "openai/gpt-oss-20b"  # bare id for the Groq batch API
LITELLM_MODEL = "groq/openai/gpt-oss-20b"  # provider-prefixed for litellm.completion
REASONING_EFFORT: ReasoningEffort = "low"

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def messages_for(text: str) -> list[dict]:
    """The chat messages used for both the interactive and batch paths."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


class Preprocessor:
    """Interactive single-item rewriter; tracks token usage and cost."""

    def __init__(
        self,
        model_name: str = LITELLM_MODEL,
        reasoning_effort: ReasoningEffort = REASONING_EFFORT,
    ):
        self.model_name = model_name
        self.reasoning_effort: ReasoningEffort = reasoning_effort
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def preprocess(self, text: str) -> str:
        response = cast(
            ModelResponse,
            completion(
                messages=messages_for(text),
                model=self.model_name,
                reasoning_effort=self.reasoning_effort,
            ),
        )
        usage = response.usage  # type: ignore[attr-defined]
        if usage:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
        self.total_cost += response._hidden_params["response_cost"]
        return response.choices[0].message.content or ""  # type: ignore[union-attr]
