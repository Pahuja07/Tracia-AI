from src.criminalNetwork.config.configuration import ConfigurationManager
from src.criminalNetwork.components.evidence_extraction import EvidenceExtraction
class EvidenceExtractionPipeline:
    def main(self): return EvidenceExtraction(ConfigurationManager().get_evidence_extraction_config()).run()
