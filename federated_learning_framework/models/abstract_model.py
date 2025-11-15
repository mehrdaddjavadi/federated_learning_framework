from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class AbstractModel(ABC):
    def __init__(self):
        self.model_config: Dict[str, Any] = {}
        self.model_architecture: str = ""
        
    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights as a list of numpy arrays."""
        pass

    @abstractmethod
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model weights from a list of numpy arrays."""
        pass

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 1) -> Dict[str, float]:
        """Train the model and return training metrics."""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        pass

    @abstractmethod
    def get_model_architecture(self) -> str:
        """Get a string representation of the model architecture."""
        pass

    @abstractmethod
    def validate_weights(self, weights: List[np.ndarray]) -> bool:
        """Validate that weights match the current model architecture."""
        pass

    def get_layer_shapes(self) -> List[Tuple[int, ...]]:
        """Get shapes of all layers in the model."""
        weights = self.get_weights()
        return [w.shape for w in weights]

    def validate_compatibility(self, other_model: 'AbstractModel') -> bool:
        """Check if two models are compatible for federated learning."""
        if not isinstance(other_model, AbstractModel):
            return False

        if self.get_model_architecture() != other_model.get_model_architecture():
            return False

        self_shapes = self.get_layer_shapes()
        other_shapes = other_model.get_layer_shapes()

        if len(self_shapes) != len(other_shapes):
            return False

        return all(s1 == s2 for s1, s2 in zip(self_shapes, other_shapes))
