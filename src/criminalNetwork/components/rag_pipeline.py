import os
import sys
from hashlib import sha256

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.criminalNetwork.components.chroma_store import create_chroma_client, get_evidence_collection
from src.criminalNetwork.entity.config_entity import RAGPipelineConfig
from src.criminalNetwork.utils.exception import CriminalNetworkException
from src.criminalNetwork.utils.logger import logger


class RAGPipeline:
    def __init__(self, config: RAGPipelineConfig):
        self.config = config
        if config.offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    def load_documents(self):
        try:
            documents = []
            for root, _, files in os.walk(self.config.input_documents_dir):
                for file_name in files:
                    if file_name.lower().endswith(".txt"):
                        docs = TextLoader(os.path.join(root, file_name), encoding="utf-8").load()
                        for doc in docs:
                            doc.metadata["source_file"] = file_name
                        documents.extend(docs)
            logger.info("Loaded %s documents from %s", len(documents), self.config.input_documents_dir)
            return documents
        except Exception as error:
            raise CriminalNetworkException(error, sys) from error

    def split_documents(self, documents):
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info("Split into %s chunks", len(chunks))
            return chunks
        except Exception as error:
            raise CriminalNetworkException(error, sys) from error

    def build_chroma_collection(self, chunks):
        """Persist evidence text, embeddings, and metadata to Chroma in batches."""
        try:
            if not chunks:
                logger.warning("No chunks to index; skipping Chroma collection build")
                return None
            client = create_chroma_client(
                self.config.chroma_mode, self.config.chroma_persist_directory,
                self.config.chroma_api_key, self.config.chroma_tenant, self.config.chroma_database,
            )
            # Indexing defines the active evidence set, so stale-case retrieval is impossible.
            try:
                client.delete_collection(self.config.chroma_collection_name)
            except Exception:
                pass
            collection = get_evidence_collection(client, self.config.chroma_collection_name)
            for start in range(0, len(chunks), self.config.chroma_batch_size):
                batch = chunks[start : start + self.config.chroma_batch_size]
                documents = [chunk.page_content for chunk in batch]
                collection.add(
                    ids=[sha256(f"{chunk.metadata.get('source', '')}:{start + index}:{chunk.page_content}".encode()).hexdigest()
                         for index, chunk in enumerate(batch)],
                    documents=documents,
                    embeddings=self.embedding_model.embed_documents(documents),
                    metadatas=[{
                        "source_file": str(chunk.metadata.get("source_file", "unknown")),
                        "source_path": str(chunk.metadata.get("source", "unknown")),
                        "chunk_index": start + index,
                    } for index, chunk in enumerate(batch)],
                )
            logger.info("Stored %s evidence chunks in Chroma collection '%s'", len(chunks), self.config.chroma_collection_name)
            return collection
        except Exception as error:
            raise CriminalNetworkException(error, sys) from error

    def run(self):
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to index")
            return
        self.build_chroma_collection(self.split_documents(documents))
        logger.info("RAG Chroma indexing completed")
