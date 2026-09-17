from src.criminalNetwork.config.configuration import ConfigurationManager
from src.criminalNetwork.components.dynamic_schema import DynamicExtractionSchema
class DynamicSchemaPipeline:
    def main(self): return DynamicExtractionSchema(ConfigurationManager().get_dynamic_schema_config()).run()
