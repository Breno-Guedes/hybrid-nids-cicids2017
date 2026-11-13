import numpy as np
import time

class FusionEngine:
    """Combina classificadores, detectores de anomalia e regras determinísticas."""
    def __init__(self):
        self.w_supervised = 0.45
        self.w_anomaly = 0.35
        self.w_rule = 0.20

        self.threshold_yellow = 0.4
        self.threshold_red = 0.75

        self.decision_cache = {}
        self.cache_max_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0

        self.decision_times = []
        self.total_decisions = 0

    @staticmethod
    def normalize_score(score):
        """Normaliza entradas inválidas, negativas ou acima de 1 para [0, 1]."""
        value = float(score)
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _cached_risk_computation(self, supervised_score, anomaly_score, rule_score):
        risk_score = (
            (self.w_supervised * supervised_score) +
            (self.w_anomaly * anomaly_score) +
            (self.w_rule * rule_score)
        )

        if rule_score > 0.8:
            risk_score = 1.0

        return round(float(np.clip(risk_score, 0.0, 1.0)), 4)

    def compute_risk_score(self, supervised_score, anomaly_score, rule_score):
        start_time = time.time()
        supervised_score = self.normalize_score(supervised_score)
        anomaly_score = self.normalize_score(anomaly_score)
        rule_score = self.normalize_score(rule_score)

        cache_key = (
            round(supervised_score, 2),
            round(anomaly_score, 2),
            round(rule_score, 2)
        )

        if cache_key in self.decision_cache:
            self.cache_hits += 1
            risk_score = self.decision_cache[cache_key]
        else:
            self.cache_misses += 1
            risk_score = self._cached_risk_computation(
                supervised_score, anomaly_score, rule_score
            )

            if len(self.decision_cache) < self.cache_max_size:
                self.decision_cache[cache_key] = risk_score

        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)
        self.total_decisions += 1

        if self.total_decisions % 1000 == 0:
            self._cleanup_cache()

        return risk_score

    def _cleanup_cache(self):
        """Remove entradas antigas quando o cache se aproxima do limite."""
        if len(self.decision_cache) > self.cache_max_size * 0.8:
            items_to_remove = int(len(self.decision_cache) * 0.2)
            keys_to_remove = list(self.decision_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.decision_cache[key]

    def get_cache_stats(self):
        """Retorna estatísticas de desempenho do cache."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        avg_decision_time = np.mean(self.decision_times) if self.decision_times else 0

        return {
            'cache_size': len(self.decision_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_decisions': self.total_decisions,
            'avg_decision_time': avg_decision_time
        }

    def determine_alert_level(self, risk_score):
        """Determina o nível Green/Yellow/Red a partir do score normalizado."""
        if risk_score > self.threshold_red:
            return "RED", "Malicious/Attack"
        elif risk_score >= self.threshold_yellow:
            return "YELLOW", "Suspicious"
        else:
            return "GREEN", "Normal"

    def process_flow(
        self,
        supervised_score,
        anomaly_score,
        rule_score,
        supervised_scores=None,
        anomaly_scores=None,
    ):
        """Processa scores individuais e retorna veredito com score limitado a 100."""
        if supervised_scores:
            values = [self.normalize_score(value) for value in supervised_scores.values()]
            supervised_score = float(np.mean(values)) if values else supervised_score
        if anomaly_scores:
            values = [self.normalize_score(value) for value in anomaly_scores.values()]
            anomaly_score = float(np.mean(values)) if values else anomaly_score

        supervised_score = self.normalize_score(supervised_score)
        anomaly_score = self.normalize_score(anomaly_score)
        rule_score = self.normalize_score(rule_score)
        if supervised_scores:
            supervised_scores = {
                name: self.normalize_score(value)
                for name, value in supervised_scores.items()
            }
        if anomaly_scores:
            anomaly_scores = {
                name: self.normalize_score(value)
                for name, value in anomaly_scores.items()
            }
        final_score = self.compute_risk_score(supervised_score, anomaly_score, rule_score)
        color, status = self.determine_alert_level(final_score)
        
        return {
            "risk_score": final_score,
            "alert_color": color,
            "status": status,
            "details": {
                "supervised_score": supervised_score,
                "anomaly_score": anomaly_score,
                "rule_score": rule_score,
                "supervised_scores": supervised_scores or {},
                "anomaly_scores": anomaly_scores or {},
            }
        }

if __name__ == "__main__":
    # Test cases
    engine = FusionEngine()
    
    # Case 1: High Attack Probability
    print("Test Case 1 (Attack):", engine.process_flow(0.9, 0.8, 0.5))
    
    # Case 2: Suspicious
    print("Test Case 2 (Suspicious):", engine.process_flow(0.3, 0.8, 0.1))
    
    # Case 3: Normal
    print("Test Case 3 (Normal):", engine.process_flow(0.1, 0.1, 0.0))
