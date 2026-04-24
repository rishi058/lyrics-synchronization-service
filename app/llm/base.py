import time
import os
import json
from helpers.config import COHERE_API_KEY, GEMINI_API_KEY
from helpers.logger import CustomLogger

MAX_RETRIES = 2
RETRY_BASE_DELAY = 5  # seconds

# from langchain_cohere import ChatCohere
# from langchain_core.prompts import ChatPromptTemplate
# LLM_MODEL = "command-a-reasoning-08-2025"

# class BaseLLM:
#     """Thin wrapper around ChatCohere with structured-output support."""

#     _client: ChatCohere | None = None  # class-level singleton

#     def __init__(self):
#         if not COHERE_API_KEY:
#             raise ValueError("COHERE_API_KEY environment variable is not set.")
#         # Reuse a single client across all instances
#         if BaseLLM._client is None:
#             BaseLLM._client = ChatCohere(
#                 model=LLM_MODEL,
#                 api_key=COHERE_API_KEY,
#                 temperature=0.2,
#             )
#         self.llm = BaseLLM._client

#     # ── single invocation (with retry) ───────────────────────────────────────

#     def invoke(self, prompt: str, response_format, user_data):
#         """Invoke the LLM with structured output parsing and automatic retry."""
#         structured_llm = self.llm.with_structured_output(response_format)

#         chat_prompt = ChatPromptTemplate([
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": "{user_data}"}
#         ])

#         chain = chat_prompt | structured_llm

#         last_err = None
#         for attempt in range(1, MAX_RETRIES + 1):
#             try:
#                 result = chain.invoke({"user_data": str(user_data)})
#                 return result
#             except Exception as e:
#                 last_err = e
#                 delay = RETRY_BASE_DELAY ** attempt
#                 ts = time.strftime('%X')
#                 print(f"[{ts}] ⚠️  LLM attempt {attempt}/{MAX_RETRIES} failed: {e}")
#                 CustomLogger.log(f"LLM retry {attempt}/{MAX_RETRIES} – {e}")
#                 if attempt < MAX_RETRIES:
#                     time.sleep(delay)

#         raise RuntimeError(f"LLM invocation failed after {MAX_RETRIES} attempts") from last_err

#     # ── chunked invocation (shared by all refiners) ──────────────────────────

#     def invoke_chunked(
#         self,
#         *,
#         items: list,
#         prompt: str,
#         response_format,
#         chunk_size: int = 120,
#         result_key: str = "words",
#         label: str = "REFINING",
#     ) -> list:
#         """
#         Split *items* into chunks, invoke the LLM for each chunk, and
#         concatenate the results.  Falls back to the original chunk data
#         when a single chunk fails.

#         Parameters
#         ----------
#         items         : flat list of dicts (or strings) to process.
#         prompt        : system prompt.
#         response_format : Pydantic model for structured output.
#         chunk_size    : max items per LLM call.
#         result_key    : attribute name on the response model that holds
#                         the list of results (e.g. "words", "refined_lyrics").
#         label         : tag used in log messages.
#         """
#         total = len(items)
#         print(f"[{time.strftime('%X')}] [{label}] {total} items → chunk_size={chunk_size}")
#         CustomLogger.log(f"--- [{label}] LLM INPUT ---\n{items}")

#         chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]
#         all_results: list = []

#         for idx, chunk in enumerate(chunks, 1):
#             ts = time.strftime('%X')
#             print(f"[{ts}] [{label}] chunk {idx}/{len(chunks)} ({len(chunk)} items)…")
#             try:
#                 result = self.invoke(prompt, response_format, chunk)
#                 part = getattr(result, result_key)
#                 all_results.extend(
#                     item.model_dump() if hasattr(item, "model_dump") else item
#                     for item in part
#                 )
#                 print(f"[{time.strftime('%X')}] [{label}] chunk {idx} ✔")
#             except Exception as e:
#                 print(f"[{time.strftime('%X')}] ⚠️ [{label}] chunk {idx} failed: {e}")
#                 print(f"[{time.strftime('%X')}] [{label}] falling back to originals for chunk {idx}")
#                 all_results.extend(chunk)

#         print(f"[{time.strftime('%X')}] [{label}] done — {len(all_results)} items out")
#         CustomLogger.log(f"--- [{label}] LLM OUTPUT ---\n{all_results}")
#         return all_results


# ── Gemini-backed BaseLLM ─────────────────────────────────────────────────────
# Uses google-genai SDK with Gemma 4 31B (instruction-tuned).
# API key is read from the GEMINI_API_KEY environment variable.

from google import genai
from google.genai import types as genai_types

GEMINI_MODEL = "gemma-4-31b-it"
# GEMINI_MODEL = "gemini-3-flash-preview"   

class BaseLLM:
    """Thin wrapper around google-genai (Gemma 4 31B) with structured-output support."""

    _client: genai.Client | None = None  # class-level singleton

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        # Reuse a single client across all instances
        if BaseLLM._client is None:
            BaseLLM._client = genai.Client(api_key=GEMINI_API_KEY)
        self.client = BaseLLM._client

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _llm_cache_dir(self) -> str:
        # base.py lives in app/llm/ → go up twice to reach app/ → cache/llm/
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(app_dir, "cache", "llm")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _load_llm_cache(self, cache_name: str):
        """Return cached list from JSON, or None if not present."""
        path = os.path.join(self._llm_cache_dir(), f"{cache_name}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_llm_cache(self, cache_name: str, data: list) -> None:
        """Persist a list to JSON for future runs."""
        path = os.path.join(self._llm_cache_dir(), f"{cache_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # ── single invocation (with retry) ───────────────────────────────────────


    def invoke(self, prompt: str, response_format, user_data):
        """Invoke the LLM with structured output parsing and automatic retry.

        Parameters
        ----------
        prompt          : System / instruction prompt string.
        response_format : Pydantic model class — its JSON schema defines the
                          desired structured output.
        user_data       : Data to embed as the user turn (converted to str).
        """
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"{prompt}\n\n{str(user_data)}")],
            )
        ]
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_format.model_json_schema(),
            # thinking_config=genai_types.ThinkingConfig(thinking_level="medium"),  #! ONLY ELIGIBLE WITH GEMINI 3 FLASH
        )

        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )
                # Gemma (and other LLMs) sometimes wrap output in ```json ... ``` fences or leave stray backticks.
                # The safest way is to slice from the first '{' or '[' to the last '}' or ']'.
                text = response.text.strip()
                start_idx = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=0)
                end_idx = max([text.rfind("}"), text.rfind("]")], default=len(text) - 1)
                
                if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                    text = text[start_idx:end_idx + 1]
                    
                return response_format.model_validate_json(text)
            except Exception as e:
                last_err = e
                delay = RETRY_BASE_DELAY ** attempt
                ts = time.strftime('%X')
                print(f"[{ts}] ⚠️  LLM attempt {attempt}/{MAX_RETRIES} failed: {e}")
                CustomLogger.log(f"LLM retry {attempt}/{MAX_RETRIES} – {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM invocation failed after {MAX_RETRIES} attempts"
        ) from last_err

    # ── chunked invocation (shared by all refiners) ──────────────────────────

    def invoke_chunked(
        self,
        *,
        items: list,
        prompt: str,
        response_format,
        chunk_size: int = 200,
        result_key: str = "words",
        label: str = "REFINING",
    ) -> list:
        """
        Split *items* into chunks, invoke the LLM for each chunk, and
        concatenate the results.  Falls back to the original chunk data
        when a single chunk fails.

        Parameters
        ----------
        items           : flat list of dicts (or strings) to process.
        prompt          : system / instruction prompt.
        response_format : Pydantic model for structured output.
        chunk_size      : max items per LLM call.
        result_key      : attribute name on the response model that holds
                          the list of results (e.g. "words", "refined_lyrics").
        label           : tag used in log messages.
        """
        total = len(items)
        print(f"[{time.strftime('%X')}] [{label}] {total} items → chunk_size={chunk_size}")
        CustomLogger.log(f"--- [{label}] LLM INPUT ---\n{items}")

        chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]
        all_results: list = []

        for idx, chunk in enumerate(chunks, 1):
            ts = time.strftime('%X')
            print(f"[{ts}] [{label}] chunk {idx}/{len(chunks)} ({len(chunk)} items)…")
            try:
                result = self.invoke(prompt, response_format, chunk)
                part = getattr(result, result_key)
                all_results.extend(
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in part
                )
                print(f"[{time.strftime('%X')}] [{label}] chunk {idx} ✔")
            except Exception as e:
                print(f"[{time.strftime('%X')}] ⚠️ [{label}] chunk {idx} failed: {e}")
                print(f"[{time.strftime('%X')}] [{label}] falling back to originals for chunk {idx}")
                all_results.extend(chunk)

        print(f"[{time.strftime('%X')}] [{label}] done — {len(all_results)} items out")
        CustomLogger.log(f"--- [{label}] LLM OUTPUT ---\n{all_results}")
        
        return all_results