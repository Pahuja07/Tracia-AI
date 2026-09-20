# src/criminalNetwork/config/configuration.py

from pathlib import Path

from src.criminalNetwork.constants import CONFIG_FILE_PATH

from src.criminalNetwork.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    EntityExtractionConfig
    , CaseUploadConfig, CaseExtractionConfig, RelationshipExtractionConfig, EntityResolutionConfig,
    GraphBuilderConfig, GraphAnalyticsConfig, EvidenceIntegrityConfig,RAGPipelineConfig,AgentConfig,
    DocumentProcessingConfig, CaseUnderstandingConfig, DynamicSchemaConfig, EvidenceExtractionConfig, TabularModelConfig,
    CrossCaseAnalysisConfig, FeatureEngineeringConfig, AssociationMiningConfig, StatisticalAnalysisConfig, LeadScoringConfig,
    SpatialIntelligenceConfig
)

from src.criminalNetwork.entity.config_entity import DataIngestionConfig, RelationshipExtractionConfig

from src.criminalNetwork.utils.common import read_yaml, create_directories

import os
from dotenv import load_dotenv

load_dotenv(override=True)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.project_root = Path(config_filepath).resolve().parent.parent
        load_dotenv(self.project_root / ".env", override=True)
        self.config = read_yaml(config_filepath)
        create_directories(self._resolve_path(self.config["artifacts_root"]))

    def _resolve_path(self, path: str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def get_data_ingestion_config(self, dataset_name: str) -> DataIngestionConfig:
        dataset_cfg = self.config["datasets"][dataset_name]

        create_directories(self._resolve_path(self.config["data_ingestion"]["processed_data_dir"]))

        return DataIngestionConfig(

            root_dir=self._resolve_path(self.config["data_ingestion"]["raw_data_dir"]),
            raw_path=self._resolve_path(dataset_cfg["raw_path"]),
            processed_path=self._resolve_path(dataset_cfg["processed_path"]),
            mapping_file=self._resolve_path(dataset_cfg["mapping_file"]),
        )

    def get_data_preprocessing_config(self, dataset_name: str) -> DataPreprocessingConfig:
        dataset_cfg = self.config["datasets"][dataset_name]
        processed_dir = self._resolve_path(self.config["data_ingestion"]["processed_data_dir"])

        create_directories(processed_dir)

        return DataPreprocessingConfig(
            root_dir=processed_dir,
            dataset_name=dataset_name,
            raw_path=self._resolve_path(dataset_cfg["raw_path"]),
            processed_path=self._resolve_path(dataset_cfg["processed_path"]),
            mapping_file=self._resolve_path(dataset_cfg["mapping_file"]),
        )

    def get_entity_extraction_config(self) -> EntityExtractionConfig:
        entity_cfg = self.config["entity_extraction"]
        root_dir = self._resolve_path(entity_cfg["root_dir"])
        output_dir = self._resolve_path(entity_cfg["output_dir"])
        processed_paths = {
            name: self._resolve_path(dataset_cfg["processed_path"])
            for name, dataset_cfg in self.config["datasets"].items()
        }

        create_directories(root_dir)
        create_directories(output_dir)

        return EntityExtractionConfig(
            root_dir=root_dir,
            processed_data_dir=self._resolve_path(
                self.config["data_ingestion"]["processed_data_dir"]
            ),
            output_dir=output_dir,
            dataset_names=list(processed_paths),
            processed_paths=processed_paths,
        )

    def get_case_upload_config(self) -> CaseUploadConfig:
        case_cfg = self.config["case_upload"]
        input_dir = self._resolve_path(case_cfg["input_dir"])
        upload_dir = self._resolve_path(case_cfg["upload_dir"])
        manifest_path = self._resolve_path(case_cfg["manifest_path"])
        create_directories(input_dir)
        create_directories(upload_dir)
        return CaseUploadConfig(input_dir, upload_dir, manifest_path)

    def get_case_extraction_config(self) -> CaseExtractionConfig:
        case_cfg = self.config["case_extraction"]
        output_dir = self._resolve_path(case_cfg["output_dir"])
        create_directories(output_dir)
        return CaseExtractionConfig(
            manifest_path=self._resolve_path(self.config["case_upload"]["manifest_path"]),
            output_dir=output_dir,
            common_entities_path=self._resolve_path(case_cfg["common_entities_path"]),
        )
    def get_document_processing_config(self) -> DocumentProcessingConfig:
        cfg = self.config["document_processing"]; output_dir = self._resolve_path(cfg["output_dir"]); create_directories(output_dir)
        return DocumentProcessingConfig(self._resolve_path(self.config["case_upload"]["manifest_path"]), output_dir, self._resolve_path(cfg["documents_path"]))
    def get_case_understanding_config(self) -> CaseUnderstandingConfig:
        cfg = self.config["case_understanding"]; output_dir = self._resolve_path(cfg["output_dir"]); create_directories(output_dir)
        return CaseUnderstandingConfig(self._resolve_path(self.config["document_processing"]["documents_path"]), output_dir, self._resolve_path(cfg["profiles_path"]))
    def get_dynamic_schema_config(self) -> DynamicSchemaConfig:
        cfg = self.config["dynamic_schema"]; output_dir = self._resolve_path(cfg["output_dir"]); create_directories(output_dir)
        return DynamicSchemaConfig(self._resolve_path(self.config["case_understanding"]["profiles_path"]), output_dir, self._resolve_path(cfg["schemas_path"]))
    def get_evidence_extraction_config(self) -> EvidenceExtractionConfig:
        cfg = self.config["evidence_extraction"]; output_dir = self._resolve_path(cfg["output_dir"]); create_directories(output_dir)
        return EvidenceExtractionConfig(self._resolve_path(self.config["document_processing"]["documents_path"]), self._resolve_path(self.config["dynamic_schema"]["schemas_path"]), output_dir, self._resolve_path(cfg["entities_path"]), self._resolve_path(cfg["evidence_path"]))
    def get_tabular_model_config(self) -> TabularModelConfig:
        cfg = self.config["tabular_model"]; output_dir = self._resolve_path(cfg["output_dir"]); create_directories(output_dir)
        evidence_cfg = self.config["evidence_extraction"]
        return TabularModelConfig(self._resolve_path(self.config["document_processing"]["documents_path"]), self._resolve_path(self.config["case_understanding"]["profiles_path"]), self._resolve_path(evidence_cfg["entities_path"]), self._resolve_path(evidence_cfg["evidence_path"]), output_dir, self._resolve_path(cfg["cases_path"]), self._resolve_path(cfg["entities_path"]), self._resolve_path(cfg["evidence_path"]))
    def get_relationship_extraction_config(self) -> RelationshipExtractionConfig:
        cfg = self.config["relationship_extraction"]
        output_path = self._resolve_path(cfg["output_path"])
        create_directories(output_path.parent)

        return RelationshipExtractionConfig(
            common_entities_path=self._resolve_path(cfg["common_entities_path"]),
            relationship_mapping_file=self._resolve_path(cfg["relationship_mapping_file"]),
            output_path=output_path,
        )

    def get_entity_resolution_config(self) -> EntityResolutionConfig:
        cfg = self.config["entity_resolution"]
        root_dir = self._resolve_path(cfg["root_dir"])
        create_directories(root_dir)
        return EntityResolutionConfig(
            root_dir=root_dir,
            input_entities_file=self._resolve_path(cfg["input_entities_file"]),
            input_relationships_file=self._resolve_path(cfg["input_relationships_file"]),
            resolved_entities_file=self._resolve_path(cfg["resolved_entities_file"]),
            resolved_relationships_file=self._resolve_path(cfg["resolved_relationships_file"]),
            entity_mapping_file=self._resolve_path(cfg["entity_mapping_file"]),
            fuzzy_threshold=cfg["fuzzy_threshold"],
        )
    def get_graph_builder_config(self) -> GraphBuilderConfig:
        config = self.config["graph_builder"]
        root_dir = self._resolve_path(config["root_dir"])
        create_directories(root_dir)
        raw_uri = os.getenv("NEO4J_URI", "")
        raw_flag = os.getenv("NEO4J_TRUST_SELF_SIGNED_CERTIFICATE", "false")
        return GraphBuilderConfig(
            root_dir=root_dir,
            entity_mapping_file=self._resolve_path(config["entity_mapping_file"]),
            resolved_entities_file=self._resolve_path(config["resolved_entities_file"]),
            resolved_relationships_file=self._resolve_path(config["resolved_relationships_file"]),
            graph_build_log_file=self._resolve_path(config["graph_build_log_file"]),
            graph_schema_file=self._resolve_path(config["graph_schema_file"]),
            neo4j_uri=raw_uri,
            neo4j_username=os.environ["NEO4J_USERNAME"],
            neo4j_password=os.environ["NEO4J_PASSWORD"],
            neo4j_database=os.getenv("NEO4J_DATABASE") or None,
            trust_self_signed_certificate=raw_flag.lower() == "true",
    )
    def get_graph_analytics_config(self) -> GraphAnalyticsConfig:
        config = self.config["graph_analytics"]
        root_dir = self._resolve_path(config["root_dir"])
        create_directories(root_dir)
        raw_uri = os.getenv("NEO4J_URI", "")
        raw_flag = os.getenv("NEO4J_TRUST_SELF_SIGNED_CERTIFICATE", "false")
        return GraphAnalyticsConfig(
            root_dir=root_dir,
            centrality_output_file=self._resolve_path(config["centrality_output_file"]),
            community_output_file=self._resolve_path(config["community_output_file"]),
            top_n_report_file=self._resolve_path(config["top_n_report_file"]),
            top_n=config["top_n"],
            neo4j_uri=raw_uri,
            neo4j_username=os.environ["NEO4J_USERNAME"],
            neo4j_password=os.environ["NEO4J_PASSWORD"],
            neo4j_database=os.getenv("NEO4J_DATABASE") or None,
            trust_self_signed_certificate=raw_flag.lower() == "true",
    )

    def get_cross_case_analysis_config(self) -> CrossCaseAnalysisConfig:
        cfg = self.config["cross_case_analysis"]; path = self._resolve_path(cfg["output_file"]); create_directories(path.parent)
        return CrossCaseAnalysisConfig(self._resolve_path(self.config["entity_resolution"]["resolved_entities_file"]), self._resolve_path(self.config["entity_resolution"]["resolved_relationships_file"]), path)
    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        cfg = self.config["feature_engineering"]; path = self._resolve_path(cfg["output_file"]); create_directories(path.parent)
        return FeatureEngineeringConfig(self._resolve_path(self.config["cross_case_analysis"]["output_file"]), self._resolve_path(self.config["entity_resolution"]["resolved_relationships_file"]), path, cfg["time_decay_days"], cfg["location_decay_km"])
    def get_association_mining_config(self) -> AssociationMiningConfig:
        cfg = self.config["association_mining"]; path = self._resolve_path(cfg["rules_output_file"]); create_directories(path.parent)
        return AssociationMiningConfig(self._resolve_path(self.config["tabular_model"]["entities_path"]), self._resolve_path(self.config["entity_resolution"]["resolved_relationships_file"]), path, cfg["min_support"], cfg["min_confidence"], cfg["lift_cap"], self._resolve_path(cfg["role_output_file"]))
    def get_statistical_analysis_config(self) -> StatisticalAnalysisConfig:
        cfg = self.config["statistical_analysis"]; correlation = self._resolve_path(cfg["correlation_output_file"]); create_directories(correlation.parent)
        return StatisticalAnalysisConfig(self._resolve_path(self.config["feature_engineering"]["output_file"]), correlation, self._resolve_path(cfg["bayesian_output_file"]))
    def get_lead_scoring_config(self) -> LeadScoringConfig:
        cfg = self.config["lead_scoring"]; path = self._resolve_path(cfg["output_file"]); create_directories(path.parent)
        return LeadScoringConfig(self._resolve_path(self.config["feature_engineering"]["output_file"]), self._resolve_path(self.config["association_mining"]["rules_output_file"]), self._resolve_path(self.config["statistical_analysis"]["bayesian_output_file"]), path, cfg["weights"])

    def get_spatial_intelligence_config(self) -> SpatialIntelligenceConfig:
        cfg = self.config["spatial_intelligence"]
        return SpatialIntelligenceConfig(
            entities_file=self._resolve_path(cfg["entities_file"]),
            evidence_file=self._resolve_path(cfg["evidence_file"]),
            relationships_file=self._resolve_path(cfg["relationships_file"]),
            resolved_entities_file=self._resolve_path(cfg["resolved_entities_file"]),
            centrality_file=self._resolve_path(cfg["centrality_file"]),
            community_file=self._resolve_path(cfg["community_file"]),
            lead_scores_file=self._resolve_path(cfg["lead_scores_file"]),
        )

    def get_evidence_integrity_config(self) -> EvidenceIntegrityConfig:
        config = self.config["evidence_integrity"]
        root_dir = self._resolve_path(config["root_dir"])
        input_documents_dir = self._resolve_path(config["input_documents_dir"])
        create_directories(root_dir)
        create_directories(input_documents_dir)
        return EvidenceIntegrityConfig(
            root_dir=root_dir,
            input_documents_dir=input_documents_dir,
            hash_ledger_file=self._resolve_path(config["hash_ledger_file"]),
            verification_report_file=self._resolve_path(config["verification_report_file"]),
            hash_algorithm=config["hash_algorithm"],
        )
    def get_rag_pipeline_config(self) -> RAGPipelineConfig:
        config = self.config["rag_pipeline"]
        root_dir = self._resolve_path(config["root_dir"])
        input_documents_dir = self._resolve_path(config["input_documents_dir"])
        vector_store_dir = self._resolve_path(config["vector_store_dir"])
        chunk_metadata_file = self._resolve_path(config["chunk_metadata_file"])
        create_directories(root_dir)
        create_directories(vector_store_dir)
        return RAGPipelineConfig(
            root_dir=root_dir,
            input_documents_dir=input_documents_dir,
            vector_store_dir=vector_store_dir,
            chunk_metadata_file=chunk_metadata_file,
            embedding_model_name=config["embedding_model_name"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            offline_mode=bool(config.get("offline_mode", True)),
    )
    def get_agent_config(self) -> AgentConfig:
        config = self.config["agent"]
        raw_uri = os.getenv("NEO4J_URI", "")
        raw_flag = os.getenv("NEO4J_TRUST_SELF_SIGNED_CERTIFICATE", "false")
        return AgentConfig(
            vector_store_dir=self._resolve_path(config["vector_store_dir"]),
            embedding_model_name=config["embedding_model_name"],
            top_suspects_file=self._resolve_path(config["top_suspects_file"]),
            centrality_file=self._resolve_path(config["centrality_file"]),
            community_file=self._resolve_path(config["community_file"]),
            llm_model_name=config["llm_model_name"],
            retrieval_k=config["retrieval_k"],
            neo4j_uri=raw_uri,
            neo4j_username=os.environ["NEO4J_USERNAME"],
            neo4j_password=os.environ["NEO4J_PASSWORD"],
            trust_self_signed_certificate=raw_flag.lower() == "true",
            openai_api_key=os.environ["OPENAI_API_KEY"],
    )



