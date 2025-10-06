"""Metrics and evaluation utilities for Legal AI Assistant."""
import logging
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    @staticmethod
    def calculate_perplexity(loss: float) -> float:
       
        return np.exp(loss)
    
    @staticmethod
    def calculate_token_accuracy(
        predictions: List[List[int]],
        references: List[List[int]],
    ) -> float:
        
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            min_len = min(len(pred), len(ref))
            total_tokens += min_len
            correct_tokens += sum(
                p == r for p, r in zip(pred[:min_len], ref[:min_len])
            )
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    @staticmethod
    def calculate_response_length_stats(
        responses: List[str]
    ) -> Dict[str, float]:
       
        lengths = [len(response.split()) for response in responses]
        
        return {
            "mean_length": np.mean(lengths),
            "median_length": np.median(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "std_length": np.std(lengths),
        }
    
    @staticmethod
    def calculate_diversity_metrics(
        responses: List[str]
    ) -> Dict[str, float]:
       
        # Calculate unique n-grams
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return set(
                ' '.join(words[i:i+n])
                for i in range(len(words) - n + 1)
            )
        
        all_unigrams = set()
        all_bigrams = set()
        all_trigrams = set()
        
        total_unigrams = 0
        total_bigrams = 0
        total_trigrams = 0
        
        for response in responses:
            unigrams = get_ngrams(response, 1)
            bigrams = get_ngrams(response, 2)
            trigrams = get_ngrams(response, 3)
            
            all_unigrams.update(unigrams)
            all_bigrams.update(bigrams)
            all_trigrams.update(trigrams)
            
            total_unigrams += len(unigrams)
            total_bigrams += len(bigrams)
            total_trigrams += len(trigrams)
        
        return {
            "distinct_1": len(all_unigrams) / max(total_unigrams, 1),
            "distinct_2": len(all_bigrams) / max(total_bigrams, 1),
            "distinct_3": len(all_trigrams) / max(total_trigrams, 1),
            "unique_unigrams": len(all_unigrams),
            "unique_bigrams": len(all_bigrams),
            "unique_trigrams": len(all_trigrams),
        }


class TrainingMetricsTracker:
    """Track and summarize training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
    
    def add_metric(self, name: str, value: float, step: int) -> None:
       
        self.metrics[name].append({
            "value": value,
            "step": step,
        })
    
    def get_latest(self, name: str) -> float:
       
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return self.metrics[name][-1]["value"]
    
    def get_average(self, name: str, last_n: int = None) -> float:
        
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = [m["value"] for m in self.metrics[name]]
        
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary of statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = [m["value"] for m in self.metrics[name]]
        
        return {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "latest": values[-1],
        }
    
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all tracked metrics.
        
        Returns:
            Dictionary of all metrics
        """
        return dict(self.metrics)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = defaultdict(list)
    
    def log_summary(self) -> None:
        """Log summary of all metrics."""
        logger.info("=" * 80)
        logger.info("Training Metrics Summary")
        logger.info("=" * 80)
        
        for name in self.metrics:
            summary = self.get_summary(name)
            logger.info(f"\n{name}:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info("=" * 80)