from src.criminalNetwork.components.tabular_model import TabularModelBuilder
from src.criminalNetwork.config.configuration import ConfigurationManager
class TabularModelPipeline:
    def main(self): return TabularModelBuilder(ConfigurationManager().get_tabular_model_config()).run()
