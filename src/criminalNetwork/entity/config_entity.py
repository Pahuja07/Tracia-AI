from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_path: Path
    processed_path: Path
    mapping_file: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    dataset_name: str
    raw_path: Path
    processed_path: Path
    mapping_file: Path


@dataclass(frozen=True)
class EntityExtractionConfig:
    root_dir: Path
    processed_data_dir: Path
    output_dir: Path
    dataset_names: list[str]
    processed_paths: dict[str, Path]


@dataclass(frozen=True)
class CaseUploadConfig:
    input_dir: Path
    upload_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class CaseExtractionConfig:
    manifest_path: Path
    output_dir: Path
    common_entities_path: Path

@dataclass(frozen=True)
class DocumentProcessingConfig:
    manifest_path: Path
    output_dir: Path
    documents_path: Path

@dataclass(frozen=True)
class CaseUnderstandingConfig:
    documents_path: Path
    output_dir: Path
    profiles_path: Path

@dataclass(frozen=True)
class DynamicSchemaConfig:
    profiles_path: Path
    output_dir: Path
    schemas_path: Path

@dataclass(frozen=True)
class EvidenceExtractionConfig:
    documents_path: Path
    schemas_path: Path
    output_dir: Path
    entities_path: Path
    evidence_path: Path

@dataclass(frozen=True)
class TabularModelConfig:
    documents_path: Path
    profiles_path: Path
    extracted_entities_path: Path
    extracted_evidence_path: Path
    output_dir: Path
    cases_path: Path
    entities_path: Path
    evidence_path: Path

@dataclass(frozen=True)
class RelationshipExtractionConfig:
    common_entities_path: Path
    relationship_mapping_file: Path
    output_path: Path

@dataclass(frozen=True)
class EntityResolutionConfig:
    root_dir: Path
    input_entities_file: Path
    input_relationships_file: Path
    resolved_entities_file: Path
    resolved_relationships_file: Path
    entity_mapping_file: Path
    fuzzy_threshold: int

@dataclass(frozen=True)
class GraphBuilderConfig:
    root_dir: Path
    entity_mapping_file: Path
    resolved_entities_file: Path
    resolved_relationships_file: Path
    graph_build_log_file: Path
    graph_schema_file: Path          # ← new field
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str | None
    trust_self_signed_certificate: bool

@dataclass(frozen=True)
class GraphAnalyticsConfig:
    root_dir: Path
    centrality_output_file: Path
    community_output_file: Path
    top_n_report_file: Path
    top_n: int
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str | None
    trust_self_signed_certificate: bool


@dataclass(frozen=True)
class EvidenceIntegrityConfig:
    root_dir: Path
    input_documents_dir: Path
    hash_ledger_file: Path
    verification_report_file: Path
    hash_algorithm: str

@dataclass(frozen=True)
class RAGPipelineConfig:
    root_dir: Path
    input_documents_dir: Path
    vector_store_dir: Path
    chunk_metadata_file: Path
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
@dataclass(frozen=True)
class AgentConfig:
    vector_store_dir: Path
    embedding_model_name: str
    top_suspects_file: Path
    centrality_file: Path
    community_file: Path
    llm_model_name: str
    retrieval_k: int
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    trust_self_signed_certificate: bool
    openai_api_key: str

@dataclass(frozen=True)
class CrossCaseAnalysisConfig:
    resolved_entities_file: Path
    resolved_relationships_file: Path
    output_file: Path

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    cross_case_file: Path
    resolved_relationships_file: Path
    output_file: Path
    time_decay_days: float
    location_decay_km: float

@dataclass(frozen=True)
class AssociationMiningConfig:
    entities_file: Path
    relationships_file: Path
    rules_output_file: Path
    min_support: float
    min_confidence: float
    lift_cap: float
    role_output_file: Path

@dataclass(frozen=True)
class StatisticalAnalysisConfig:
    features_file: Path
    correlation_output_file: Path
    bayesian_output_file: Path

@dataclass(frozen=True)
class LeadScoringConfig:
    features_file: Path
    association_rules_file: Path
    bayesian_file: Path
    output_file: Path
    weights: dict


@dataclass(frozen=True)
class SpatialIntelligenceConfig:
    """Read-only inputs for the map/API integration layer."""
    entities_file: Path
    evidence_file: Path
    relationships_file: Path
    resolved_entities_file: Path
    centrality_file: Path
    community_file: Path
    lead_scores_file: Path

