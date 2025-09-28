
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import numpy as np
from pathlib import Path
import os

from services.qdrant import QdrantService
from config import cfg


class Loader:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        self.qdrant = QdrantService()
        self.data_dir = Path(cfg.DATA_DIR)

    def load_local_pdf(self):
        pdf_dir = self.data_dir / "pdf"
        files_path = self.get_dir_files(pdf_dir)
        data = []
        for file_path in files_path:
            docs = self.get_documents_from_file(file_path)
            data.extend(docs)

        self.qdrant.load_data(data)


    def get_documents_from_file(self, file_path: list[str]) -> list[Document]:
        loader = PyPDFLoader(file_path)
        data = []
        for page in loader.lazy_load():
            self.text_splitter.split_text(page.page_content)
            pages = [Document(page_content=text) for text in self.text_splitter.split_text(page.page_content)]
            data.extend(pages)
        
        return data

    def get_dir_files(self, directory_path: str) -> list[str]:
        all_files = []

        for root, directories, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        return all_files