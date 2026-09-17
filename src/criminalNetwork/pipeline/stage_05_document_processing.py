from src.criminalNetwork.config.configuration import ConfigurationManager
from src.criminalNetwork.components.document_processor import DocumentProcessor
class DocumentProcessingPipeline:
    def main(self): return DocumentProcessor(ConfigurationManager().get_document_processing_config()).run()
