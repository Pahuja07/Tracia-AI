import os
import sys
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.criminalNetwork.entity.config_entity import RAGPipelineConfig
from src.criminalNetwork.utils.logger import logger
from src.criminalNetwork.utils.exception import CriminalNetworkException


class RAGPipeline:
    def __init__(self, config: RAGPipelineConfig):
        self.config = config
        if self.config.offline_mode:
            # Reuse the cached model. This avoids remote Hub checks and makes
            # indexing deterministic in constrained/offline deployments.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.chunk_metadata = []

    def load_documents(self):
        """input_documents_dir se saari .txt files load karta hai."""
        try:
            documents = []
            for root, _, files in os.walk(self.config.input_documents_dir):
                for file_name in files:
                    if not file_name.lower().endswith(".txt"):
                        continue
                    file_path = os.path.join(root, file_name)
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_file"] = file_name
                    documents.extend(docs)

            logger.info(f"Loaded {len(documents)} documents from {self.config.input_documents_dir}")
            return documents
        except Exception as e:
                raise CriminalNetworkException(e, sys)

    def split_documents(self, documents):
        """Documents ko chunks mein todta hai (embedding ke liye manageable size)."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks (chunk_size={self.config.chunk_size})")
            return chunks
        except Exception as e:
                raise CriminalNetworkException(e, sys)

    def build_vector_store(self, chunks):
        """Chunks ko embed karke FAISS vector store banata hai aur disk pe save karta hai."""
        try:
            if not chunks:
                logger.warning("No chunks to index — skipping vector store build")
                return None

            vector_store = FAISS.from_documents(chunks, self.embedding_model)
            vector_store.save_local(str(self.config.vector_store_dir))
            logger.info(f"Vector store saved to {self.config.vector_store_dir}")

            for i, chunk in enumerate(chunks):
                self.chunk_metadata.append({
                    "chunk_id": i,
                    "source_file": chunk.metadata.get("source_file"),
                    "chunk_preview": chunk.page_content[:100].replace("\n", " "),
                })

            metadata_df = pd.DataFrame(self.chunk_metadata)
            metadata_df.to_csv(self.config.chunk_metadata_file, index=False)
            logger.info(f"Chunk metadata saved to {self.config.chunk_metadata_file}")

            return vector_store
        except Exception as e:
                raise CriminalNetworkException(e, sys)

    def run(self):
        try:
            logger.info("Starting RAG pipeline stage (indexing)")

            documents = self.load_documents()
            if not documents:
                logger.warning("No documents found to index")
                return

            chunks = self.split_documents(documents)
            self.build_vector_store(chunks)

            logger.info("RAG pipeline stage (indexing) completed")
        except Exception as e:
                raise CriminalNetworkException(e, sys)
