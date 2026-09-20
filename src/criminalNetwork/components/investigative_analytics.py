"""Cross-case, association, statistical, and deterministic lead-scoring analytics."""
import math
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from src.criminalNetwork.entity.config_entity import CrossCaseAnalysisConfig, FeatureEngineeringConfig, AssociationMiningConfig, StatisticalAnalysisConfig, LeadScoringConfig


class CrossCaseAnalysis:
    def __init__(self, config: CrossCaseAnalysisConfig): self.config = config
    def run(self):
        entities = pd.read_csv(self.config.resolved_entities_file)
        rels = pd.read_csv(self.config.resolved_relationships_file).dropna(subset=["source_entity_id", "target_entity_id"])
        links = []
        for entity_id, occurrences in pd.concat([rels[["case_id", "source_entity_id"]].rename(columns={"source_entity_id": "entity_id"}), rels[["case_id", "target_entity_id"]].rename(columns={"target_entity_id": "entity_id"})]).drop_duplicates().groupby("entity_id"):
            cases = sorted(occurrences.case_id.astype(str).unique())
            if len(cases) > 1: links.append({"entity_id": entity_id, "related_cases": "|".join(cases), "related_case_count": len(cases)})
        result = pd.DataFrame(links).merge(entities, on="entity_id", how="left") if links else pd.DataFrame(columns=["entity_id", "related_cases", "related_case_count"])
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True); result.to_csv(self.config.output_file, index=False); return result


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig): self.config = config
    def run(self):
        rels = pd.read_csv(self.config.resolved_relationships_file).dropna(subset=["source_entity_id", "target_entity_id"]).copy()
        recurrence = pd.read_csv(self.config.cross_case_file) if self.config.cross_case_file.exists() else pd.DataFrame()
        maximum = max(recurrence.related_case_count.max(), 1) if not recurrence.empty else 1
        lookup = recurrence.set_index("entity_id").related_case_count.to_dict() if not recurrence.empty else {}
        # Missing dates/coordinates deliberately yield neutral 0, never invented proximity.
        rels["entity_relationship_score"] = 100.0
        rels["crime_similarity_score"] = 0.0
        rels["time_proximity_score"] = 0.0
        rels["location_proximity_score"] = 0.0
        rels["cross_case_recurrence_score"] = rels.apply(lambda row: 100 * max(lookup.get(row.source_entity_id, 1), lookup.get(row.target_entity_id, 1)) / maximum, axis=1)
        raw_confidence = rels["extraction_confidence"] if "extraction_confidence" in rels else pd.Series(0.5, index=rels.index)
        rels["evidence_confidence"] = pd.to_numeric(raw_confidence, errors="coerce").fillna(0.5) * 100
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True); rels.to_csv(self.config.output_file, index=False); return rels


class AssociationMining:
    def __init__(self, config: AssociationMiningConfig): self.config = config
    def run(self):
        entities = pd.read_csv(self.config.entities_file)
        transactions = [[f"{r.entity_type}:{r.entity_value}" for r in group.itertuples(index=False)] for _, group in entities.groupby("case_id")]
        if not transactions: result = pd.DataFrame(); self.config.rules_output_file.parent.mkdir(parents=True, exist_ok=True); result.to_csv(self.config.rules_output_file, index=False); return result
        matrix = TransactionEncoder().fit(transactions).transform(transactions)
        # Pairwise rules are the most interpretable investigator-facing
        # associations and avoid exponential itemset growth in dense cases.
        frequent = fpgrowth(pd.DataFrame(matrix, columns=TransactionEncoder().fit(transactions).columns_), min_support=self.config.min_support, use_colnames=True, max_len=2)
        if frequent.empty or len(frequent) < 2: result = pd.DataFrame()
        else:
            result = association_rules(frequent, metric="confidence", min_threshold=self.config.min_confidence)
            result["normalized_lift"] = result["lift"].clip(upper=self.config.lift_cap) / self.config.lift_cap
            result["association_strength_score"] = 100 * (result["support"] + result["confidence"] + result["normalized_lift"]) / 3
            result = result.sort_values("association_strength_score", ascending=False).head(5000)
        self.config.rules_output_file.parent.mkdir(parents=True, exist_ok=True); result.to_csv(self.config.rules_output_file, index=False)
        # Associative role mining: roles are labels for observable graph patterns, never assertions about criminality.
        relationships = pd.read_csv(self.config.relationships_file).dropna(subset=["source_entity_id", "target_entity_id"])
        graph = nx.Graph(); graph.add_edges_from(relationships[["source_entity_id", "target_entity_id"]].itertuples(index=False, name=None))
        degree = dict(graph.degree()); bridge = nx.betweenness_centrality(graph) if graph else {}
        recurrence = pd.concat([relationships.source_entity_id, relationships.target_entity_id]).value_counts().to_dict()
        roles = []
        for entity_id in graph.nodes:
            role = "peripheral_participant"
            if bridge.get(entity_id, 0) >= 0.1: role = "bridge_entity"
            elif degree.get(entity_id, 0) >= max(degree.values(), default=1) * 0.75: role = "high_degree_connector"
            elif recurrence.get(entity_id, 0) > 1: role = "recurring_entity"
            roles.append({"entity_id": entity_id, "observable_role": role, "degree": degree.get(entity_id, 0), "betweenness": bridge.get(entity_id, 0), "observed_relationship_count": recurrence.get(entity_id, 0)})
        pd.DataFrame(roles).to_csv(self.config.role_output_file, index=False)
        return result


class StatisticalAnalysis:
    def __init__(self, config: StatisticalAnalysisConfig): self.config = config
    def run(self):
        df = pd.read_csv(self.config.features_file)
        numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
        rows = [{"feature_a": a, "feature_b": b, "spearman_rho": spearmanr(numeric[a], numeric[b]).statistic, "p_value": spearmanr(numeric[a], numeric[b]).pvalue} for i, a in enumerate(numeric.columns) for b in numeric.columns[i + 1:]]
        correlations = pd.DataFrame(rows)
        # Bayesian network target is relationship support, never guilt/criminality.
        confidence = pd.to_numeric(df["evidence_confidence"] if "evidence_confidence" in df else pd.Series(0, index=df.index), errors="coerce").fillna(0) / 100
        recurrence = pd.to_numeric(df["cross_case_recurrence_score"] if "cross_case_recurrence_score" in df else pd.Series(0, index=df.index), errors="coerce").fillna(0) / 100
        training = pd.DataFrame({"evidence_high": (confidence >= .5).astype(int), "recurrence_high": (recurrence >= .5).astype(int)})
        training["supported"] = ((training.evidence_high == 1) & (training.recurrence_high == 1)).astype(int)
        model = DiscreteBayesianNetwork([("evidence_high", "supported"), ("recurrence_high", "supported")])
        estimator = BayesianEstimator(model, training)
        model.add_cpds(*estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=5))
        inference = VariableElimination(model)
        posterior = []
        for evidence_high, recurrence_high in zip(training.evidence_high, training.recurrence_high):
            query = inference.query(["supported"], evidence={"evidence_high": int(evidence_high), "recurrence_high": int(recurrence_high)}, show_progress=False)
            states = query.state_names["supported"]
            posterior.append(float(query.values[states.index(1)]) if 1 in states else 0.0)
        bayesian = df[["case_id", "source_entity_id", "target_entity_id"]].copy(); bayesian["relationship_support_posterior"] = posterior; bayesian["hypothesis"] = "documented_relationship_supported"
        self.config.correlation_output_file.parent.mkdir(parents=True, exist_ok=True); correlations.to_csv(self.config.correlation_output_file, index=False); bayesian.to_csv(self.config.bayesian_output_file, index=False); return correlations, bayesian


class LeadScoring:
    def __init__(self, config: LeadScoringConfig): self.config = config
    def run(self):
        df = pd.read_csv(self.config.features_file); weights = self.config.weights
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6): raise ValueError("Lead-score weights must sum to 1")
        rules = pd.read_csv(self.config.association_rules_file) if self.config.association_rules_file.exists() else pd.DataFrame()
        sas = float(rules.association_strength_score.mean()) if not rules.empty else 0.0
        df["association_strength_score"] = sas
        mapping = {"entity_relationship": "entity_relationship_score", "crime_similarity": "crime_similarity_score", "time_proximity": "time_proximity_score", "location_proximity": "location_proximity_score", "cross_case_recurrence": "cross_case_recurrence_score", "association_strength": "association_strength_score"}
        df["lead_match_score"] = sum(weights[key] * df[column].fillna(0) for key, column in mapping.items())
        df["score_version"] = "lms-v1"; self.config.output_file.parent.mkdir(parents=True, exist_ok=True); df.to_csv(self.config.output_file, index=False); return df
