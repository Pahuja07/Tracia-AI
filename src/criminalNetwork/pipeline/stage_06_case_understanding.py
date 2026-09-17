from src.criminalNetwork.config.configuration import ConfigurationManager
from src.criminalNetwork.components.case_understanding import CaseUnderstanding
class CaseUnderstandingPipeline:
    def main(self): return CaseUnderstanding(ConfigurationManager().get_case_understanding_config()).run()
