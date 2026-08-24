from abc import ABC, abstractmethod
from typing import Generator, Optional
import os
import requests
import json
import warnings

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Generates a streaming response for the given prompt.

        Args:
            prompt: The user input or full prompt context.
            system_prompt: Optional system instruction (if supported separately).
            temperature: Optional sampling temperature override -- None
                (default) uses the provider/model's own default. Added for
                callers needing deterministic output (e.g. an LLM-as-judge
                scoring pass, v4-plan.md Stage 6), not used by normal chat.

        Yields:
            Chunks of the generated text response.
        """
        pass

class MockProvider(LLMProvider):
    """Echo provider for testing purposes."""

    def generate(
        self, prompt: str, system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        # Mimic a stream
        response = f"[Mock Response] You said: {prompt}..."
        for word in response.split():
            yield word + " "

class OllamaProvider(LLMProvider):
    """
    Production provider using Ollama's REST API.

    Environment Variables:
        AEON_OLLAMA_HOST: Base URL (default: http://localhost:11434)
        AEON_LLM_MODEL: Model name (default: llama3)
        AEON_LLM_NUM_CTX: Fixes `options.num_ctx` to this exact value for
            every request, overriding the automatic sizing below. Unset by
            default -- set this only for reproducibility (e.g. comparing two
            runs at an identical, pinned context window).
        AEON_LLM_TIMEOUT_SECONDS: Per-request timeout in seconds passed to
            `requests.post` (default: 120). The previous hardcoded 30s was
            sized for small/fast models -- a larger local model (13B+) can
            legitimately take longer than that to emit its first token on a
            long prompt, which reads as "Could not connect" even though
            nothing is actually wrong.

    num_ctx sizing (bug found and fixed, v4-plan.md Stage 7): a fixed
    `num_ctx=8192` default silently truncated any prompt longer than that --
    Ollama does not error or warn, it just drops the excess and the model
    answers from whatever remained. Measured directly against a real Stage 7
    `full_session`-expansion prompt: 12,795 actual prompt tokens sent against
    `num_ctx=8192` -- about 36% of the intended context never reached the
    model, on a model (`gemma4:31b-cloud`) whose real context length is
    262,144 tokens (`ollama show`). This is a standing correctness gap for
    any caller handing Aeon's retrieved memory to an LLM -- "carrying real
    retrieved memory context is Aeon's whole purpose" -- not specific to one
    benchmark. Fixed by sizing `num_ctx` from the actual prompt length (via
    `_estimate_tokens()`, a conservative chars-per-token heuristic that
    undercounts rather than overcounts, since undercounting only makes this
    provider ask for a still-safely-large `num_ctx`), capped at the model's
    own advertised context length (queried once via `/api/show` and cached
    for this provider instance's lifetime -- adds one extra request on first
    use, not per generation call). If even the model's own max can't hold
    the estimated prompt, that's real, unavoidable truncation -- surfaced as
    an explicit `warnings.warn()` instead of silently happening.
    """

    # Conservative: measured ~4.2-4.3 real chars/token on actual retrieved-
    # memory prompts (English chat text); deliberately using a lower ratio
    # so the token estimate this divides out is an OVERestimate of how many
    # tokens the prompt needs -- safer to request a larger num_ctx than
    # actually required than to under-request and truncate.
    _CHARS_PER_TOKEN_ESTIMATE = 3.5
    # Headroom for the model's own generated response, which also consumes
    # the context window in Ollama's `num_ctx` accounting.
    _RESPONSE_TOKEN_HEADROOM = 1024
    _MIN_NUM_CTX = 8192

    def __init__(self):
        self.host = os.getenv("AEON_OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("AEON_LLM_MODEL", "llama3")
        self._num_ctx_override = os.getenv("AEON_LLM_NUM_CTX")
        self.timeout_seconds = float(os.getenv("AEON_LLM_TIMEOUT_SECONDS", "120"))
        self.api_url = f"{self.host}/api/generate"
        self._model_max_context: Optional[int] = None
        self._model_max_context_queried = False

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self._CHARS_PER_TOKEN_ESTIMATE)

    def _query_model_max_context(self) -> Optional[int]:
        """Queries and caches the model's real max context length via
        Ollama's `/api/show`. Returns None (not 0) if the query fails or the
        field isn't present -- callers must treat None as "unknown," not as
        "no limit," so a failed query doesn't accidentally suppress the
        truncation warning below."""
        if self._model_max_context_queried:
            return self._model_max_context
        self._model_max_context_queried = True
        try:
            r = requests.post(f"{self.host}/api/show", json={"model": self.model}, timeout=10)
            r.raise_for_status()
            model_info = r.json().get("model_info", {})
            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    self._model_max_context = int(value)
                    break
        except (requests.exceptions.RequestException, ValueError, KeyError):
            pass
        return self._model_max_context

    def _compute_num_ctx(self, prompt: str, system_prompt: str) -> int:
        if self._num_ctx_override is not None:
            return int(self._num_ctx_override)

        needed = self._estimate_tokens(prompt) + self._estimate_tokens(system_prompt) + self._RESPONSE_TOKEN_HEADROOM
        num_ctx = max(self._MIN_NUM_CTX, needed)

        model_max = self._query_model_max_context()
        if model_max is not None and num_ctx > model_max:
            warnings.warn(
                f"Prompt needs an estimated ~{needed} tokens, exceeding "
                f"'{self.model}''s own max context length ({model_max}). "
                f"Truncating to {model_max} -- Ollama will silently drop the "
                f"excess with no further warning from this point on."
            )
            num_ctx = model_max
        return num_ctx

    def generate(
        self, prompt: str, system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        # Exposed so a caller (a benchmark harness, most immediately) can
        # record what was actually sent -- v4-plan.md Stage 7 lost real
        # time to a result file with no recorded model/num_ctx, forcing a
        # dig through scratch logs to even know which model a "ceiling"
        # number came from.
        self.last_num_ctx = self._compute_num_ctx(prompt, system_prompt)
        options = {"num_ctx": self.last_num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": options,
        }

        try:
            with requests.post(
                self.api_url, json=payload, stream=True,
                timeout=self.timeout_seconds,
            ) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if not line:
                        continue
                        
                    decoded_line = line.decode("utf-8")
                    try:
                        data = json.loads(decoded_line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            yield f"\n[System Error: Could not connect to LLM Provider at {self.host}. Details: {str(e)}]"
