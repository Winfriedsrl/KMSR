from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import load_env


class LLM:
    def __init__(self, model: str):
        load_env()
        self.llm = ChatOpenAI(model=model, timeout=120)

    def ask(self, prompt: str, system_prompt: str | None = None):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return (response.content or "").strip()
