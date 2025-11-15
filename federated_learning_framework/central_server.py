import asyncio
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Set, Optional, Tuple
import time
from federated_learning_framework.connection import ConnectionServer
from websockets.exceptions import ConnectionClosedError

# Handle optional dependencies
try:
    from federated_learning_framework.encryption import create_context
    ENCRYPTION_AVAILABLE = True
except ImportError:
    create_context = lambda: None
    ENCRYPTION_AVAILABLE = False

try:
    from federated_learning_framework.privacy import DifferentialPrivacy
    PRIVACY_AVAILABLE = True
except ImportError:
    DifferentialPrivacy = None
    PRIVACY_AVAILABLE = False

try:
    from federated_learning_framework.client_selection import ClientSelector
    CLIENT_SELECTION_AVAILABLE = True
except ImportError:
    ClientSelector = None
    CLIENT_SELECTION_AVAILABLE = False

try:
    from federated_learning_framework.error_recovery import (
        ErrorRecoveryManager, ErrorContext, ErrorType
    )
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ErrorRecoveryManager = None
    ErrorContext = None
    ErrorType = None
    ERROR_RECOVERY_AVAILABLE = False

try:
    from federated_learning_framework.evaluation import ModelEvaluator, EvaluationMetrics
    EVALUATION_AVAILABLE = True
except ImportError:
    ModelEvaluator = None
    EvaluationMetrics = None
    EVALUATION_AVAILABLE = False

try:
    from federated_learning_framework.progress_tracking import (
        ProgressTracker, TrainingMetrics
    )
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    ProgressTracker = None
    TrainingMetrics = None
    PROGRESS_TRACKING_AVAILABLE = False

class CentralServer:
    def __init__(self, connection_type='websocket', host='0.0.0.0', port=8089, context=None, 
                 min_clients=2, min_available_clients=0.7, min_eval_clients=2,
                 epsilon=1.0, delta=1e-5, clip_threshold=1.0,
                 straggler_timeout=5.0,
                 output_dir: str = "fl_training"):
        self.model_weights = None
        self.lock = asyncio.Lock()
        self.clients: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        self.connection = ConnectionServer(connection_type, host, port, self.handle_client)
        self.context = context or create_context()
        
        # FedAvg parameters
        self.min_clients = min_clients
        self.min_available_clients = min_available_clients
        self.min_eval_clients = min_eval_clients
        self.client_weights: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.current_round = 0
        self.training_history = []
        
        # Privacy mechanism
        self.privacy = DifferentialPrivacy(
            epsilon=epsilon,
            delta=delta,
            clip_threshold=clip_threshold
        )
        self.privacy_metrics = {
            'epsilon_spent': 0.0,
            'delta_spent': 0.0
        }
        
        # Client selection strategy
        self.client_selector = ClientSelector(
            min_clients=min_clients,
            selection_fraction=min_available_clients,
            straggler_timeout=straggler_timeout
        )
        
        # Track client response times
        self.client_start_times: Dict[str, float] = {}
        
        # Error recovery system
        self.error_manager = ErrorRecoveryManager()
        self.client_status: Dict[str, str] = {}
        
        # Model evaluation
        self.evaluator = ModelEvaluator()
        self.validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.best_metrics: Optional[EvaluationMetrics] = None
        self.convergence_threshold = 0.001
        self.evaluation_window = 5
        
        # Progress tracking
        self.progress_tracker = None
        if PROGRESS_TRACKING_AVAILABLE:
            self.progress_tracker = ProgressTracker(output_dir)
        self.last_report_time = time.time()
        self.report_interval = 300  # Generate report every 5 minutes

    async def run_server(self):
        self.logger.info("Central Server is starting...")
        await self.connection.start()

    async def handle_client(self, websocket, client_id):
        """Handle client connection and messages with error recovery."""
        try:
            # Initialize client status
            self.client_status[client_id] = "connecting"
            
            # Wait for initial model architecture message
            try:
                message = await asyncio.wait_for(
                    self.connection.receive(client_id),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                await self._handle_error(
                    ErrorContext(
                        error_type=ErrorType.TIMEOUT_ERROR,
                        client_id=client_id,
                        round_number=self.current_round,
                        timestamp=time.time(),
                        details="Initial connection timeout"
                    )
                )
                return

            if not isinstance(message, dict) or 'model_architecture' not in message:
                await self._handle_error(
                    ErrorContext(
                        error_type=ErrorType.VALIDATION_ERROR,
                        client_id=client_id,
                        round_number=self.current_round,
                        timestamp=time.time(),
                        details="Invalid initial message format"
                    )
                )
                return

            # Validate model architecture
            client_architecture = message['model_architecture']
            if self.model_weights is not None:
                if not self.validate_model_compatibility(client_architecture):
                    await self._handle_error(
                        ErrorContext(
                            error_type=ErrorType.VALIDATION_ERROR,
                            client_id=client_id,
                            round_number=self.current_round,
                            timestamp=time.time(),
                            details="Incompatible model architecture"
                        )
                    )
                    return

            # Successfully connected
            self.clients.add(client_id)
            self.client_status[client_id] = "connected"
            self.logger.info(f"Central Server: Client {client_id} connected with valid model")

            while True:
                try:
                    message = await self.connection.receive(client_id)
                    if isinstance(message, dict):
                        if 'weights' in message and 'num_samples' in message:
                            weights = message['weights']
                            if self.validate_weights_format(weights):
                                # Save checkpoint before weight update
                                self.error_manager.save_checkpoint(
                                    self.current_round,
                                    self.model_weights,
                                    {'client_weights': self.client_weights.copy()}
                                )
                                
                                await self.transmit_weights(
                                    client_id, 
                                    weights,
                                    message['num_samples'],
                                    message.get('metrics', {})
                                )
                            else:
                                await self._handle_error(
                                    ErrorContext(
                                        error_type=ErrorType.VALIDATION_ERROR,
                                        client_id=client_id,
                                        round_number=self.current_round,
                                        timestamp=time.time(),
                                        details="Invalid weights format"
                                    )
                                )
                        elif 'data_request' in message:
                            data = await self.get_data_from_client(client_id)
                            if data is not None:
                                await self.send_data_to_client(client_id, {'data': data})

                except asyncio.TimeoutError:
                    await self._handle_error(
                        ErrorContext(
                            error_type=ErrorType.TIMEOUT_ERROR,
                            client_id=client_id,
                            round_number=self.current_round,
                            timestamp=time.time(),
                            details="Message receive timeout"
                        )
                    )
                except Exception as e:
                    await self._handle_error(
                        ErrorContext(
                            error_type=ErrorType.UNKNOWN_ERROR,
                            client_id=client_id,
                            round_number=self.current_round,
                            timestamp=time.time(),
                            details=str(e)
                        )
                    )

        except ConnectionClosedError:
            await self._handle_error(
                ErrorContext(
                    error_type=ErrorType.CONNECTION_ERROR,
                    client_id=client_id,
                    round_number=self.current_round,
                    timestamp=time.time(),
                    details="Client disconnected"
                )
            )
        finally:
            self._cleanup_client(client_id)

    def _cleanup_client(self, client_id: str):
        """Clean up client-related data structures."""
        self.clients.discard(client_id)
        if client_id in self.client_weights:
            del self.client_weights[client_id]
        if client_id in self.client_start_times:
            del self.client_start_times[client_id]
        if client_id in self.client_status:
            del self.client_status[client_id]

    async def _handle_error(self, error_context: ErrorContext) -> bool:
        """Handle errors with recovery attempts."""
        async def recovery_callback(ctx: ErrorContext) -> bool:
            try:
                if ctx.error_type == ErrorType.CONNECTION_ERROR:
                    # Wait for client to reconnect
                    return await self._wait_for_reconnection(ctx.client_id)
                    
                elif ctx.error_type == ErrorType.TIMEOUT_ERROR:
                    # Try to re-establish communication
                    return await self._retry_communication(ctx.client_id)
                    
                elif ctx.error_type == ErrorType.VALIDATION_ERROR:
                    # Restore from last checkpoint
                    return await self._restore_checkpoint(ctx.round_number)
                    
                elif ctx.error_type == ErrorType.TRAINING_ERROR:
                    # Skip problematic client for this round
                    return await self._skip_client_round(ctx.client_id)
                    
                return False
                
            except Exception as e:
                self.logger.error(f"Recovery failed: {str(e)}")
                return False

        return await self.error_manager.handle_error(error_context, recovery_callback)

    async def _wait_for_reconnection(self, client_id: str) -> bool:
        """Wait for a client to reconnect."""
        timeout = 30.0  # 30 seconds timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if client_id in self.clients and self.client_status.get(client_id) == "connected":
                return True
            await asyncio.sleep(1.0)
            
        return False

    async def _retry_communication(self, client_id: str) -> bool:
        """Attempt to re-establish communication with a client."""
        try:
            # Send ping message
            await self.send_data_to_client(client_id, {'type': 'ping'})
            
            # Wait for response
            response = await asyncio.wait_for(
                self.connection.receive(client_id),
                timeout=5.0
            )
            
            return response and isinstance(response, dict) and response.get('type') == 'pong'
        except Exception:
            return False

    async def _restore_checkpoint(self, round_number: int) -> bool:
        """Restore system state from a checkpoint."""
        model_state, round_state = self.error_manager.load_checkpoint(round_number)
        if model_state is None or round_state is None:
            return False
            
        self.model_weights = model_state
        self.client_weights = round_state['client_weights']
        return True

    async def _skip_client_round(self, client_id: str) -> bool:
        """Skip a problematic client for the current round."""
        if client_id in self.client_weights:
            del self.client_weights[client_id]
        if client_id in self.client_start_times:
            del self.client_start_times[client_id]
        return True

    def validate_model_compatibility(self, client_architecture: str) -> bool:
        """Validate that client model architecture is compatible with server model."""
        if not hasattr(self, 'model_architecture'):
            self.model_architecture = client_architecture
            return True
        return self.model_architecture == client_architecture

    def validate_weights_format(self, weights: List[np.ndarray]) -> bool:
        """Validate that weights have correct format and shapes."""
        try:
            if not isinstance(weights, list):
                return False
            if not all(isinstance(w, np.ndarray) for w in weights):
                return False
            if not hasattr(self, 'expected_shapes'):
                self.expected_shapes = [w.shape for w in weights]
                return True
            return all(w.shape == shape 
                     for w, shape in zip(weights, self.expected_shapes))
        except Exception:
            return False

    async def aggregate_weights(self) -> List[np.ndarray]:
        """Implement FedAvg algorithm with differential privacy and evaluation."""
        if not self.client_weights:
            return self.model_weights

        try:
            # Calculate total number of samples across all clients
            total_samples = sum(metadata['num_samples'] 
                            for metadata in self.client_weights.values())

            # Initialize aggregated weights with zeros
            first_client_weights = next(iter(self.client_weights.values()))['weights']
            aggregated_weights = [np.zeros_like(layer) for layer in first_client_weights]

            # Weighted average based on number of samples
            for client_data in self.client_weights.values():
                # Clip gradients for privacy
                clipped_weights = self.privacy.clip_gradients(client_data['weights'])
                
                # Calculate weight based on number of samples
                weight = client_data['num_samples'] / total_samples
                
                # Add weighted contribution
                for i, layer in enumerate(clipped_weights):
                    aggregated_weights[i] += layer * weight

            # Add noise for differential privacy
            noisy_weights = self.privacy.add_noise(aggregated_weights, total_samples)

            # Update privacy metrics
            privacy_spent = self.privacy.get_privacy_spent(self.current_round)
            self.privacy_metrics.update(privacy_spent)
            
            # Evaluate model if validation data is available
            if self.validation_data is not None:
                await self._evaluate_global_model(noisy_weights)
            
            # Check for convergence
            convergence_status = self.evaluator.get_convergence_status(
                threshold=self.convergence_threshold,
                window_size=self.evaluation_window
            )
            
            if convergence_status['converged']:
                self.logger.info("Model has converged! Stopping training.")
                # You might want to implement a mechanism to notify clients
                
            # Log metrics
            self.logger.info(
                f"Round {self.current_round} - "
                f"Privacy budget spent - ε: {self.privacy_metrics['epsilon_spent']:.4f}, "
                f"δ: {self.privacy_metrics['delta_spent']:.4e}"
            )

            return noisy_weights
        except Exception as e:
            self.logger.error(f"Error in weight aggregation: {str(e)}")
            return self.model_weights  # Return previous weights on error

    async def _evaluate_global_model(self, weights: List[np.ndarray]):
        """Evaluate the global model and update progress tracking."""
        try:
            # Create a temporary model for evaluation
            temp_model = self._create_temp_model()
            temp_model.set_weights(weights)
            
            # Evaluate model
            x_val, y_val = self.validation_data
            metrics = self.evaluator.evaluate_model(
                temp_model, x_val, y_val, self.current_round
            )
            
            # Update best metrics if needed
            if self.best_metrics is None or metrics.accuracy > self.best_metrics.accuracy:
                self.best_metrics = metrics
                self.error_manager.save_checkpoint(
                    self.current_round,
                    weights,
                    {'metrics': metrics}
                )
            
            # Get convergence status
            convergence_status = self.evaluator.get_convergence_status(
                threshold=self.convergence_threshold,
                window_size=self.evaluation_window
            )
            
            # Update progress tracking if available
            if PROGRESS_TRACKING_AVAILABLE and self.progress_tracker:
                # Create training metrics
                training_metrics = TrainingMetrics(
                    round_number=self.current_round,
                    accuracy=metrics.accuracy,
                    loss=metrics.loss,
                    client_metrics={
                        client_id: {
                            'accuracy': data.get('metrics', {}).get('accuracy', 0.0),
                            'loss': data.get('metrics', {}).get('loss', 0.0)
                        }
                        for client_id, data in self.client_weights.items()
                    },
                    privacy_budget=self.privacy_metrics,
                    convergence_metrics=convergence_status
                )
                
                # Update progress tracking and generate report
                self.progress_tracker.update_metrics(training_metrics)
                current_time = time.time()
                if current_time - self.last_report_time >= self.report_interval:
                    report = self.progress_tracker.generate_training_report()
                    self.logger.info(f"\n{report}")
                    self.last_report_time = current_time
            
            # Log basic evaluation results
            self.logger.info(
                f"Round {self.current_round} Evaluation:\n"
                f"Accuracy: {metrics.accuracy:.4f}\n"
                f"Loss: {metrics.loss:.4f}\n"
                f"F1 Score: {metrics.f1_score:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating global model: {str(e)}")

    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress summary."""
        if PROGRESS_TRACKING_AVAILABLE and self.progress_tracker:
            return self.progress_tracker.get_training_summary()
        return {
            'current_round': self.current_round,
            'num_clients': len(self.clients),
            'privacy_metrics': self.privacy_metrics
        }
        
    def get_client_progress(self, client_id: str) -> Dict[str, Any]:
        """Get training progress for a specific client."""
        client_stats = {
            'status': self.client_status.get(client_id, 'unknown'),
            'error_stats': self.error_manager.get_client_status(client_id)
        }
        if PROGRESS_TRACKING_AVAILABLE and self.progress_tracker:
            client_stats['performance'] = self.progress_tracker.client_progress.get(client_id, [])
        return client_stats

    def set_validation_data(self, x_val: np.ndarray, y_val: np.ndarray):
        """Set validation data for model evaluation."""
        self.validation_data = (x_val, y_val)
        
    def _create_temp_model(self) -> Any:
        """Create a temporary model for evaluation."""
        # This should be implemented based on your model architecture
        raise NotImplementedError(
            "You must implement _create_temp_model in your subclass"
        )

    async def transmit_weights(self, client_id: str, weights: List[np.ndarray], 
                           num_samples: int, metrics: Dict[str, float] = None):
        """Handle weight updates from clients and coordinate FedAvg rounds."""
        async with self.lock:
            # Calculate response time and update client metrics
            if client_id in self.client_start_times:
                response_time = time.time() - self.client_start_times[client_id]
                if metrics is None:
                    metrics = {}
                self.client_selector.update_client_metrics(
                    client_id, metrics, response_time
                )

            self.client_weights[client_id] = {
                'weights': weights,
                'num_samples': num_samples
            }

            # Select clients for the next round if we have enough updates
            selected_clients, scores = self.client_selector.select_clients(
                self.clients, self.current_round
            )
            
            if len(self.client_weights) >= len(selected_clients):
                # Aggregate weights using FedAvg
                self.model_weights = await self.aggregate_weights()
                self.current_round += 1
                
                # Clear client weights for next round
                self.client_weights.clear()
                
                # Record start times for selected clients
                current_time = time.time()
                self.client_start_times = {
                    cid: current_time for cid in selected_clients
                }
                
                # Broadcast new weights to selected clients
                round_info = {
                    'weights': self.model_weights,
                    'round': self.current_round,
                    'client_score': scores.get(client_id, 0.0)
                }
                
                await asyncio.gather(*[
                    self.connection.send(cid, round_info) 
                    for cid in selected_clients
                ])
                
                # Log round completion and client selection info
                self.logger.info(
                    f"Round {self.current_round} completed with "
                    f"{len(selected_clients)} selected clients"
                )
                
                # Log straggler information
                stragglers = [cid for cid in self.clients 
                            if self.client_selector.is_straggler(cid)]
                if stragglers:
                    self.logger.warning(
                        f"Detected {len(stragglers)} stragglers in round "
                        f"{self.current_round}"
                    )

    async def send_data_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to a specific client with proper error handling."""
        try:
            self.logger.info(f"Central Server: Sending data to client {client_id}")
            await self.connection.send(client_id, data)
        except Exception as e:
            self.logger.error(f"Error sending data to client {client_id}: {str(e)}")

    async def get_data_from_client(self, client_id: str) -> np.ndarray:
        """Get data from a specific client with proper validation."""
        try:
            self.logger.info(f"Central Server: Requesting data from client {client_id}")
            # In a real implementation, this would fetch actual data from clients
            data = np.random.rand(10, 3072)  # Placeholder
            if not isinstance(data, np.ndarray):
                raise ValueError("Invalid data format received from client")
            return data
        except Exception as e:
            self.logger.error(f"Error getting data from client {client_id}: {str(e)}")
            return None

    def query_active_learning(self, unlabeled_data: np.ndarray, model: Any) -> np.ndarray:
        """Select most informative samples using enhanced active learning strategy."""
        try:
            if not hasattr(self, 'active_learning_strategy'):
                from federated_learning_framework.active_learning import ActiveLearningStrategy
                self.active_learning_strategy = ActiveLearningStrategy(
                    uncertainty_weight=0.7,
                    diversity_weight=0.3,
                    batch_size=10
                )

            # Use the advanced active learning strategy
            selected_indices, scores = self.active_learning_strategy.select_samples(
                model=model,
                unlabeled_data=unlabeled_data,
                feature_extractor=getattr(model, 'extract_features', None)
            )
            
            # Log selection metrics
            self.logger.info(f"Selected {len(selected_indices)} samples for active learning")
            self.logger.info(f"Average uncertainty score: {scores['uncertainty'].mean():.4f}")
            self.logger.info(f"Average diversity score: {scores['diversity'].mean():.4f}")
            
            return selected_indices
            
        except Exception as e:
            self.logger.error(f"Error in active learning query: {str(e)}")
            return np.array([])
