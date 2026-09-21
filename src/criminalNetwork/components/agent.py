import os
import sys
import httpx
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase

from src.criminalNetwork.entity.config_entity import AgentConfig
from src.criminalNetwork.utils.logger import logger
from src.criminalNetwork.utils.exception import CriminalNetworkException
from src.criminalNetwork.utils.common import normalize_neo4j_uri


class CriminalNetworkAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured. Add it to .env before using the chatbot.")
        # The pipeline already downloaded this model while building FAISS. Do
        # not block investigator queries on a remote Hugging Face check.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={"local_files_only": True},
        )
        self.vector_store = self._load_vector_store()
        self.last_answer_mode = "llm"
        use_env_proxy = os.getenv("OPENAI_USE_ENV_PROXY", "false").lower() == "true"
        self.http_client = httpx.Client(timeout=httpx.Timeout(45.0, connect=15.0), trust_env=use_env_proxy)
        self.llm = ChatOpenAI(
            model=self.config.llm_model_name,
            api_key=self.config.openai_api_key,
            http_client=self.http_client,
            max_retries=1,
        )
        uri = normalize_neo4j_uri(self.config.neo4j_uri, self.config.trust_self_signed_certificate)
        self.driver = GraphDatabase.driver(
            uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )

    def close(self):
        self.driver.close()
        self.http_client.close()

    def _load_vector_store(self):
        try:
            return FAISS.load_local(
                str(self.config.vector_store_dir),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def retrieve_context(self, query: str) -> str:
        """RAG: query se relevant case document chunks nikalta hai."""
        try:
            docs = self.vector_store.similarity_search(query, k=self.config.retrieval_k)
            context = "\n\n".join(
                f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )
            return context
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def get_top_suspects(self, n: int = 5) -> pd.DataFrame:
        """graph_analytics ke output se top-N influential entities deta hai."""
        try:
            df = pd.read_csv(self.config.top_suspects_file)
            return df.head(n)
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def query_entity_connections(self, entity_name: str) -> str:
        """Neo4j se ek entity ke saare direct connections nikalta hai."""
        try:
            query = (
                "MATCH (a:Entity)-[r]-(b:Entity) "
                "WHERE toLower(a.name) CONTAINS toLower($name) "
                "RETURN a.name AS entity, type(r) AS relationship, b.name AS connected_to, "
                "b.entity_type AS connected_type"
            )
            with self.driver.session() as session:
                results = list(session.run(query, name=entity_name))

            if not results:
                return f"No connections found for '{entity_name}' in the graph."

            lines = [
                f"{r['entity']} --[{r['relationship']}]--> {r['connected_to']} ({r['connected_type']})"
                for r in results
            ]
            return "\n".join(lines)
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def _build_prompt(self, user_query: str, rag_context: str, graph_context: str, suspects_context: str) -> str:
        return f"""You are an investigative assistant helping analyze a criminal network graph.
Answer the investigator's question using ONLY the context provided below. Be precise and factual.
If the context doesn't contain enough information, say so clearly.

=== RELEVANT CASE DOCUMENTS ===
{rag_context if rag_context else "No relevant case documents found."}

=== GRAPH CONNECTIONS ===
{graph_context if graph_context else "No graph connections found."}

=== TOP INFLUENTIAL ENTITIES (by network analysis) ===
{suspects_context if suspects_context else "No analytics data available."}

=== INVESTIGATOR'S QUESTION ===
{user_query}

Answer:"""

    def query_case_connections(self, case_id: str) -> str:
        """Retrieve only recorded Neo4j links for a selected case."""
        query = (
            "MATCH (a:Entity)-[r]-(b:Entity) "
            "WHERE a.name = $case_id OR b.name = $case_id "
            "RETURN a.name AS entity, type(r) AS relationship, b.name AS connected_to, "
            "b.entity_type AS connected_type LIMIT 100"
        )
        try:
            with self.driver.session() as session:
                rows = list(session.run(query, case_id=case_id))
            return "\n".join(
                f"{row['entity']} --[{row['relationship']}]--> {row['connected_to']} ({row['connected_type']})"
                for row in rows
            ) or f"No documented Neo4j relationships found for case '{case_id}'."
        except Exception as error:
            raise CriminalNetworkException(error, sys) from error

    @staticmethod
    def _fallback_answer(rag_context: str, graph_context: str, suspects_context: str) -> str:
        return (
            "LLM explanation is temporarily unavailable. The following is direct, evidence-grounded "
            "retrieval from TRACIA records, not a generated conclusion.\n\n"
            f"=== Retrieved evidence ===\n{rag_context[:3000] or 'No matching indexed evidence found.'}\n\n"
            f"=== Documented Neo4j relationships ===\n{graph_context[:2500] or 'No selected graph context.'}\n\n"
            f"=== Existing network analytics ===\n{suspects_context[:1500] or 'No analytics available.'}"
        )

    def answer_query(self, user_query: str, entity_focus: str = None, case_id: str = None) -> str:
        """Main entry point — RAG + Graph + Analytics combine karke LLM se answer generate karta hai."""
        try:
            logger.info(f"Agent received query: {user_query}")

            rag_context = self.retrieve_context(user_query)

            graph_context = ""
            if entity_focus:
                graph_context = self.query_entity_connections(entity_focus)
            elif case_id:
                graph_context = self.query_case_connections(case_id)

            suspects_df = self.get_top_suspects(n=5)
            suspects_context = suspects_df.to_string(index=False) if not suspects_df.empty else ""

            prompt = self._build_prompt(user_query, rag_context, graph_context, suspects_context)
            try:
                response = self.llm.invoke(prompt)
                logger.info("Agent generated response successfully")
                return response.content
            except Exception as error:
                logger.warning("LLM request failed; returning evidence/Neo4j fallback: %s", error)
                self.last_answer_mode = "evidence_graph_fallback"
                return self._fallback_answer(rag_context, graph_context, suspects_context)
        except Exception as e:
            raise CriminalNetworkException(e, sys)
