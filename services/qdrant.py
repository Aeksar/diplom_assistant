from qdrant_client import QdrantClient, models
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from uuid import uuid4

from config import cfg


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(url=cfg.QDRANT_URL)
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS_MODEL)
        self.collection_name = "diplom"
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                )
            )
            
    def load_data(self, pages: list[Document]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(
                id=str(uuid4()),
                payload={
                    "page_content": page.page_content
                },
                vector=self.embeddings.embed_query(page.page_content)
            )
                for page in pages
            ]
        )

    def retrive(self, query: str):
        relevant_docs = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.embeddings.embed_query(query),
            limit=10
        )
        return_set = []
        for doc in relevant_docs:
            doc_contetn = doc.payload["page_content"]
            return_set.append(Document(page_content=doc_contetn))

        return return_set