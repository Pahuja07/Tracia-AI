import sys
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
import community as community_louvain  # python-louvain package

from src.criminalNetwork.entity.config_entity import GraphAnalyticsConfig
from src.criminalNetwork.utils.logger import logger
from src.criminalNetwork.utils.exception import CriminalNetworkException
from src.criminalNetwork.utils.common import normalize_neo4j_uri


class GraphAnalytics:
    def __init__(self, config: GraphAnalyticsConfig):
        self.config = config
        uri = normalize_neo4j_uri(self.config.neo4j_uri, self.config.trust_self_signed_certificate)
        if self.config.trust_self_signed_certificate and ".neo4j.io" not in uri.lower():
            logger.warning("Neo4j self-signed certificate trust is enabled")
        self.driver = GraphDatabase.driver(
            uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )
        self.graph = nx.Graph()          # undirected — centrality/community ke liye
        self.node_attrs = {}

    def close(self):
        self.driver.close()

    def _fetch_nodes_tx(self, tx):
        query = (
            "MATCH (e:Entity) WHERE e.entity_id IS NOT NULL "
            "RETURN e.entity_id AS entity_id, e.name AS name, e.entity_type AS entity_type"
        )
        return list(tx.run(query))

    def _fetch_relationships_tx(self, tx):
        query = (
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL "
            "RETURN a.entity_id AS source, b.entity_id AS target, type(r) AS rel_type"
        )
        return list(tx.run(query))

    def load_graph_from_neo4j(self):
        """Neo4j se poora graph nikal ke networkx.Graph mein load karta hai."""
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                nodes = session.execute_read(self._fetch_nodes_tx)
                rels = session.execute_read(self._fetch_relationships_tx)

            for record in nodes:
                entity_id = record["entity_id"]
                self.graph.add_node(entity_id)
                self.node_attrs[entity_id] = {
                    "name": record["name"],
                    "entity_type": record["entity_type"],
                }

            for record in rels:
                self.graph.add_edge(record["source"], record["target"], rel_type=record["rel_type"])

            logger.info(f"Loaded graph from Neo4j: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def compute_centrality(self) -> pd.DataFrame:
        """Degree, betweenness, aur PageRank centrality compute karta hai."""
        try:
            if self.graph.number_of_nodes() == 0:
                logger.warning("Graph is empty — skipping centrality computation")
                return pd.DataFrame()

            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            pagerank = nx.pagerank(self.graph)

            records = []
            for node_id in self.graph.nodes():
                attrs = self.node_attrs.get(node_id, {})
                records.append({
                    "entity_id": node_id,
                    "name": attrs.get("name"),
                    "entity_type": attrs.get("entity_type"),
                    "degree_centrality": round(degree_centrality.get(node_id, 0), 4),
                    "betweenness_centrality": round(betweenness_centrality.get(node_id, 0), 4),
                    "pagerank": round(pagerank.get(node_id, 0), 4),
                })

            centrality_df = pd.DataFrame(records).sort_values(by="pagerank", ascending=False)
            centrality_df.to_csv(self.config.centrality_output_file, index=False)
            logger.info(f"Centrality scores saved to {self.config.centrality_output_file}")
            return centrality_df
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def compute_communities(self) -> pd.DataFrame:
        """Louvain algorithm se community/cluster detection karta hai (gang groupings)."""
        try:
            if self.graph.number_of_nodes() == 0:
                logger.warning("Graph is empty — skipping community detection")
                return pd.DataFrame()

            partition = community_louvain.best_partition(self.graph)

            records = []
            for node_id, community_id in partition.items():
                attrs = self.node_attrs.get(node_id, {})
                records.append({
                    "entity_id": node_id,
                    "name": attrs.get("name"),
                    "entity_type": attrs.get("entity_type"),
                    "community_id": community_id,
                })

            community_df = pd.DataFrame(records).sort_values(by="community_id")
            community_df.to_csv(self.config.community_output_file, index=False)
            logger.info(f"Community clusters saved to {self.config.community_output_file}")

            num_communities = community_df["community_id"].nunique()
            logger.info(f"Detected {num_communities} communities/clusters in the network")
            return community_df
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def generate_top_suspects_report(self, centrality_df: pd.DataFrame, community_df: pd.DataFrame):
        """Top-N most influential entities ka combined report banata hai (centrality + community)."""
        try:
            if centrality_df.empty:
                logger.warning("Centrality data empty — skipping top suspects report")
                return

            top_df = centrality_df.head(self.config.top_n).copy()

            if not community_df.empty:
                top_df = top_df.merge(
                    community_df[["entity_id", "community_id"]], on="entity_id", how="left"
                )

            top_df.to_csv(self.config.top_n_report_file, index=False)
            logger.info(f"Top {self.config.top_n} suspects report saved to {self.config.top_n_report_file}")
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def write_scores_back_to_neo4j(self, centrality_df: pd.DataFrame, community_df: pd.DataFrame):
        """Centrality aur community scores ko Neo4j node properties mein wapas likhta hai."""
        try:
            if centrality_df.empty:
                return

            merged_df = centrality_df
            if not community_df.empty:
                merged_df = centrality_df.merge(
                    community_df[["entity_id", "community_id"]], on="entity_id", how="left"
                )

            with self.driver.session(database=self.config.neo4j_database) as session:
                rows = [{
                    "entity_id": row["entity_id"],
                    "degree_centrality": float(row["degree_centrality"]),
                    "betweenness_centrality": float(row["betweenness_centrality"]),
                    "pagerank": float(row["pagerank"]),
                    "community_id": int(row["community_id"]) if pd.notna(row.get("community_id")) else -1,
                } for _, row in merged_df.iterrows()]
                session.run(
                    "UNWIND $rows AS row MATCH (e:Entity {entity_id: row.entity_id}) "
                    "SET e.degree_centrality=row.degree_centrality, "
                    "e.betweenness_centrality=row.betweenness_centrality, "
                    "e.pagerank=row.pagerank, e.community_id=row.community_id",
                    rows=rows,
                ).consume()

            logger.info("Centrality and community scores written back to Neo4j nodes")
        except Exception as e:
            raise CriminalNetworkException(e, sys)

    def run(self):
        try:
            logger.info("Starting graph analytics stage")

            self.load_graph_from_neo4j()
            centrality_df = self.compute_centrality()
            community_df = self.compute_communities()
            self.generate_top_suspects_report(centrality_df, community_df)
            self.write_scores_back_to_neo4j(centrality_df, community_df)

            logger.info("Graph analytics stage completed")
        except Exception as e:
            raise CriminalNetworkException(e, sys)
        finally:
            self.close()
