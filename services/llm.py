from langchain import hub
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate

from states.graph_states import State
from config.settings import MISTRAL_API_KEY, LLM_MODELS, PROMPT_MODELS

class LLMService:
    def __init__(self):
        self.prompt: PromptTemplate = hub.pull(PROMPT_MODELS)
        self.model = ChatMistralAI(
            model=LLM_MODELS,
            temperature=0,
            mistral_api_key=MISTRAL_API_KEY
        )

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.model.invoke(messages)
        return {"answer": response.content}