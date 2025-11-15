import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import logging

class ActiveLearningStrategy:
    def __init__(self, 
                 uncertainty_weight: float = 0.7,
                 diversity_weight: float = 0.3,
                 batch_size: int = 10):
        """Initialize Active Learning Strategy.
        
        Args:
            uncertainty_weight: Weight for uncertainty sampling (between 0 and 1)
            diversity_weight: Weight for diversity sampling (between 0 and 1)
            batch_size: Number of samples to select in each query
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.selected_history = []

    def entropy_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate entropy-based uncertainty scores."""
        entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        return entropy

    def margin_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate margin-based uncertainty scores."""
        sorted_probs = np.sort(predictions, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return -margins  # Negative margin so higher means more uncertain

    def calculate_diversity(self, features: np.ndarray) -> np.ndarray:
        """Calculate diversity scores using clustering and distance-based measures."""
        try:
            # Use k-means clustering to identify diverse samples
            n_clusters = min(self.batch_size, len(features))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate distance to cluster centers
            distances_to_centers = cdist(features, kmeans.cluster_centers_)
            
            # Calculate distance to already selected samples if any
            if self.selected_history:
                selected_features = features[self.selected_history]
                distances_to_selected = cdist(
                    features, 
                    selected_features
                ).min(axis=1)
            else:
                distances_to_selected = np.zeros(len(features))
            
            # Combine both measures
            diversity_scores = (distances_to_centers.min(axis=1) + 
                              distances_to_selected)
            
            return diversity_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating diversity scores: {str(e)}")
            return np.zeros(len(features))

    def select_samples(self, 
                      model: Any,
                      unlabeled_data: np.ndarray,
                      feature_extractor: Any = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Select most informative samples using both uncertainty and diversity.
        
        Args:
            model: The trained model for making predictions
            unlabeled_data: Unlabeled samples to select from
            feature_extractor: Optional function to extract features for diversity
            
        Returns:
            Tuple of (selected indices, scores dictionary)
        """
        try:
            # Get predictions
            predictions = model.predict(unlabeled_data)
            
            # Calculate uncertainty scores using both methods
            entropy_scores = self.entropy_uncertainty(predictions)
            margin_scores = self.margin_uncertainty(predictions)
            
            # Normalize uncertainty scores
            uncertainty_scores = (entropy_scores + margin_scores) / 2
            uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / \
                               (uncertainty_scores.max() - uncertainty_scores.min() + 1e-10)
            
            # Calculate diversity scores
            if feature_extractor is not None:
                features = feature_extractor(unlabeled_data)
            else:
                features = unlabeled_data.reshape(len(unlabeled_data), -1)
            
            diversity_scores = self.calculate_diversity(features)
            diversity_scores = (diversity_scores - diversity_scores.min()) / \
                             (diversity_scores.max() - diversity_scores.min() + 1e-10)
            
            # Combine scores
            final_scores = (self.uncertainty_weight * uncertainty_scores +
                          self.diversity_weight * diversity_scores)
            
            # Select top samples
            n_select = min(self.batch_size, len(predictions))
            selected = np.argsort(final_scores)[-n_select:]
            
            # Update selection history
            self.selected_history.extend(selected.tolist())
            
            # Return selected indices and all scores for analysis
            scores = {
                'uncertainty': uncertainty_scores,
                'diversity': diversity_scores,
                'final': final_scores
            }
            
            return selected, scores
            
        except Exception as e:
            self.logger.error(f"Error selecting samples: {str(e)}")
            # Fallback to simple uncertainty sampling
            predictions = model.predict(unlabeled_data)
            n_select = min(self.batch_size, len(predictions))
            selected = np.argsort(predictions.max(axis=1))[-n_select:]
            return selected, {}
