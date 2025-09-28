from langchain import hub
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate

from states.graph_states import State
from config.settings import cfg

class LLMService:
    def __init__(self):
        self.prompt: PromptTemplate = hub.pull(cfg.PROMPT_MODELS)
        self.model = ChatMistralAI(
            model=cfg.LLM_MODELS,
            temperature=0,
            mistral_api_key=cfg.MISTRAL_API_KEY
        )

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.model.invoke(messages)
        return response
