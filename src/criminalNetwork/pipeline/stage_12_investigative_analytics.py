from src.criminalNetwork.config.configuration import ConfigurationManager
from src.criminalNetwork.components.investigative_analytics import CrossCaseAnalysis, FeatureEngineering, AssociationMining, StatisticalAnalysis, LeadScoring
class InvestigativeAnalyticsPipeline:
    def main(self):
        manager = ConfigurationManager()
        CrossCaseAnalysis(manager.get_cross_case_analysis_config()).run()
        FeatureEngineering(manager.get_feature_engineering_config()).run()
        AssociationMining(manager.get_association_mining_config()).run()
        StatisticalAnalysis(manager.get_statistical_analysis_config()).run()
        return LeadScoring(manager.get_lead_scoring_config()).run()
