import os
import time
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
LLM_API_KEY = os.getenv("COHERE_API_KEY")
LLM_MODEL = "command-a-03-2025"


class BaseLLM:
    def __init__(self):
        if not LLM_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is not set.")
        self.llm = ChatCohere(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
        )

    def invoke(self, prompt: str, response_format, user_data):
        """Invoke the LLM with structured output parsing."""
        structured_llm = self.llm.with_structured_output(response_format)

        chat_prompt = ChatPromptTemplate([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "{user_data}"}
        ])

        chain = chat_prompt | structured_llm

        try:
            result = chain.invoke({"user_data": str(user_data)})
            return result
        except Exception as e:
            print(f"[{time.strftime('%X')}] ⚠️  LLM invocation failed: {e}")
            raise