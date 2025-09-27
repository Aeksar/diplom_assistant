import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from services.qdrant import QdrantService


class Loader:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )


    def load_local_data(self):
        file_path = "./data/diplom_data.pdf"
        loader = PyPDFLoader(file_path)
        qdrant = QdrantService()
        data = []
        for page in loader.lazy_load():
            self.text_splitter.split_text(page.page_content)
            pages = [Document(page_content=text) for text in self.text_splitter.split_text(page.page_content)]
            data.extend(pages)
        qdrant.load_data(data)


if __name__ == "__main__":
    Loader().load_local_data()