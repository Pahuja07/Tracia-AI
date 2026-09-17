from src.criminalNetwork.utils.logger import logger

from src.criminalNetwork.pipeline.stage_06_5_graph_loading import GraphBuilderPipeline
from src.criminalNetwork.pipeline.stage_01_bulk_ingestion import DataIngestionPipeline
from src.criminalNetwork.pipeline.stage_02_normalize_datasets import DataPreprocessingPipeline
from src.criminalNetwork.pipeline.stage_03_extraction import EntityExtractionPipeline
from src.criminalNetwork.pipeline.stage_04_case_upload import CaseUploadPipeline
from src.criminalNetwork.pipeline.stage_05_document_processing import DocumentProcessingPipeline
from src.criminalNetwork.pipeline.stage_06_case_understanding import CaseUnderstandingPipeline
from src.criminalNetwork.pipeline.stage_07_dynamic_schema import DynamicSchemaPipeline
from src.criminalNetwork.pipeline.stage_08_evidence_extraction import EvidenceExtractionPipeline
from src.criminalNetwork.pipeline.stage_09_tabular_model import TabularModelPipeline
from src.criminalNetwork.pipeline.stage_06_relationship import RelationshipExtractionPipeline
from src.criminalNetwork.pipeline.stage_06_4_entity_resolution import EntityResolutionPipeline
from src.criminalNetwork.pipeline.stage_07_05_graph_analytics import GraphAnalyticsPipeline
from src.criminalNetwork.pipeline.stage_09_evidence_hash import EvidenceIntegrityPipeline
from src.criminalNetwork.pipeline.stage_12_investigative_analytics import InvestigativeAnalyticsPipeline
from src.criminalNetwork.pipeline.stage_07_rag_setup import RAGIndexingPipeline
# ---------------- Stage 01: Data Ingestion ----------------
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()  # aapke existing ingestion loop/logic ke hisaab se
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ---------------- Stage 02: Data Preprocessing ----------------
STAGE_NAME = "Data Preprocessing Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_preprocessing = DataPreprocessingPipeline()
    data_preprocessing.main()  # sabhi 5 datasets normalise karega
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Entity Extraction Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    entity_extraction = EntityExtractionPipeline()
    entity_extraction.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ---------------- Stage 04: Case Upload ----------------
STAGE_NAME = "Case Upload Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    case_upload = CaseUploadPipeline()
    case_upload.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ---------------- Stage 05: Document Processing ----------------
STAGE_NAME = "Document Processing Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    DocumentProcessingPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Case Understanding and Dynamic Schema Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    CaseUnderstandingPipeline().main()
    DynamicSchemaPipeline().main()
    EvidenceExtractionPipeline().main()
    TabularModelPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Relationship Extraction Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    relationship_extraction = RelationshipExtractionPipeline()
    relationship_extraction.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Entity Resolution Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EntityResolutionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Graph Builder Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = GraphBuilderPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Graph Analytics Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = GraphAnalyticsPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Cross-Case, Association, Statistics and Lead Scoring Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    InvestigativeAnalyticsPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evidence Integrity Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvidenceIntegrityPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "RAG Indexing Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = RAGIndexingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
