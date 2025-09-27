from langchain_core.documents import Document
from services.qdrant import QdrantService
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from services import LLMService, QdrantService, Loader
from states.graph_states import State




def build_graph():
    qdrant = QdrantService()
    llm = LLMService()

    def retrieve(state: State):
        retrieved_docs = qdrant.retrive(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = llm.prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.model.invoke(messages)
        return {"answer": response.content}
    

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def main():
    graph = build_graph()
    Loader().load_local_data()

    while True:
        question = input("Ask your quetion about diplom: ")
        result = graph.invoke({"question": question})
        print(result["answer"])
        print('--------------------------')
    

if __name__ == "__main__":
    main()

