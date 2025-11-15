"""
Federated Learning Framework

A professional, modular and extensible framework for federated learning applications
with privacy preservation, security features, and error recovery.

Features:
    - FedAvg aggregation with configurable strategies
    - Differential privacy with gradient clipping
    - Active learning sample selection
    - Client-side privacy preservation
    - Homomorphic encryption support (optional)
    - Comprehensive error recovery
    - Model evaluation framework
    - Progress tracking and visualization
    - Connection management with heartbeat monitoring
    - Support for TensorFlow and PyTorch models
    
Example:
    Basic federated learning setup:
    
    >>> from federated_learning_framework import CentralServer, ClientDevice
    >>> import asyncio
    >>> 
    >>> async def main():
    ...     server = CentralServer(port=8089, min_clients=2)
    ...     await server.run_server()
    >>> 
    >>> asyncio.run(main())
"""

__version__ = "1.0.1"
__author__ = "Mehrdad Javadi"
__email__ = "mehrdaddjavadi@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/mehrdaddjavadi/federated_learning_framework"

# Core components
from .active_learning import ActiveLearningStrategy
from .central_server import CentralServer
from .client_device import ClientDevice
from .connection import ConnectionServer, ConnectionClient
from .decorators import federated_learning_decorator, encryption_decorator
from .utils import setup_logging

# Models
try:
    from .models.tensorflow_model import TensorFlowModel
except (ImportError, ModuleNotFoundError):
    TensorFlowModel = None

try:
    from .models.pytorch_model import PyTorchModel
except (ImportError, ModuleNotFoundError):
    PyTorchModel = None

# Encryption (optional)
try:
    from .encryption import create_context, encrypt_weights, decrypt_weights
except (ImportError, ModuleNotFoundError):
    create_context = None
    encrypt_weights = lambda ctx, w: w
    decrypt_weights = lambda ctx, w: w

# Privacy (optional)
try:
    from .privacy import DifferentialPrivacy
except (ImportError, ModuleNotFoundError):
    DifferentialPrivacy = None

# Client selection (optional)
try:
    from .client_selection import ClientSelector
except (ImportError, ModuleNotFoundError):
    ClientSelector = None

# Error recovery (optional)
try:
    from .error_recovery import ErrorRecoveryManager, ErrorContext, ErrorType
except (ImportError, ModuleNotFoundError):
    ErrorRecoveryManager = None
    ErrorContext = None
    ErrorType = None

# Evaluation (optional)
try:
    from .evaluation import ModelEvaluator, EvaluationMetrics
except (ImportError, ModuleNotFoundError):
    ModelEvaluator = None
    EvaluationMetrics = None

# Progress tracking (optional)
try:
    from .progress_tracking import ProgressTracker, TrainingMetrics
except (ImportError, ModuleNotFoundError):
    ProgressTracker = None
    TrainingMetrics = None

# Connection manager (optional)
try:
    from .connection_manager import ConnectionManager, ClientSession, ConnectionState
except (ImportError, ModuleNotFoundError):
    ConnectionManager = None
    ClientSession = None
    ConnectionState = None

# Public API
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    
    # Core
    "CentralServer",
    "ClientDevice",
    "ConnectionServer",
    "ConnectionClient",
    "ActiveLearningStrategy",
    
    # Models
    "TensorFlowModel",
    "PyTorchModel",
    
    # Encryption
    "create_context",
    "encrypt_weights",
    "decrypt_weights",
    
    # Privacy
    "DifferentialPrivacy",
    
    # Client selection
    "ClientSelector",
    
    # Error recovery
    "ErrorRecoveryManager",
    "ErrorContext",
    "ErrorType",
    
    # Evaluation
    "ModelEvaluator",
    "EvaluationMetrics",
    
    # Progress tracking
    "ProgressTracker",
    "TrainingMetrics",
    
    # Connection manager
    "ConnectionManager",
    "ClientSession",
    "ConnectionState",
    
    # Decorators
    "federated_learning_decorator",
    "encryption_decorator",
    
    # Utilities
    "setup_logging",
]

def get_version() -> str:
    """Get framework version."""
    return __version__

def info() -> dict:
    """Get framework information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
    }

