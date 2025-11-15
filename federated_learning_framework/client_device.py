# import asyncio
# import logging
# import tensorflow as tf
# from federated_learning_framework.encryption import encrypt_weights, decrypt_weights
# from federated_learning_framework.models.tensorflow_model import TensorFlowModel
# import websockets

# class ClientDevice:
#     def __init__(self, client_id, model: TensorFlowModel, context):
#         self.client_id = client_id
#         self.model = model
#         self.context = context
#         self.connection = None
#         self.logger = logging.getLogger(__name__)

#     async def connect_to_central_server(self, uri):
#         try:
#             self.connection = await websockets.connect(uri)
#             await self.connection.send({'client_id': self.client_id})
#             self.logger.info(f"Client {self.client_id}: Connected to central server at {uri}")
#         except Exception as e:
#             self.logger.error(f"Client {self.client_id}: Error connecting to central server: {e}")

#     async def federated_learning(self, x_train, y_train):
#         try:
#             while True:
#                 weights = await self.receive_weights()
#                 if weights is None:
#                     break
#                 self.model.set_weights(decrypt_weights(self.context, weights))
#                 self.model.train(x_train, y_train, epochs=1)
#                 new_weights = self.model.get_weights()
#                 await self.send_weights(encrypt_weights(self.context, new_weights))
#         except Exception as e:
#             self.logger.error(f"Client {self.client_id}: Error in federated learning loop: {e}")

#     async def receive_weights(self):
#         try:
#             message = await self.connection.recv()
#             self.logger.info(f"Client {self.client_id}: Received weights")
#             return message['weights']
#         except Exception as e:
#             self.logger.error(f"Client {self.client_id}: Error receiving weights: {e}")

#     async def send_weights(self, weights):
#         try:
#             await self.connection.send({'weights': weights})
#             self.logger.info(f"Client {self.client_id}: Sent weights to central server")
#         except Exception as e:
#             self.logger.error(f"Client {self.client_id}: Error sending weights: {e}")

#     async def request_data(self):
#         try:
#             await self.connection.send({'data_request': True})
#             data = await self.connection.recv()
#             self.logger.info(f"Client {self.client_id}: Received data from central server")
#             return data['data']
#         except Exception as e:
#             self.logger.error(f"Client {self.client_id}: Error requesting data: {e}")


import asyncio
import logging
import websockets
import pickle

try:
    import tensorflow as tf
    from federated_learning_framework.models.tensorflow_model import TensorFlowModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TensorFlowModel = None
    TENSORFLOW_AVAILABLE = False

try:
    from federated_learning_framework.encryption import encrypt_weights, decrypt_weights
    ENCRYPTION_AVAILABLE = True
except ImportError:
    encrypt_weights = lambda ctx, w: w
    decrypt_weights = lambda ctx, w: w
    ENCRYPTION_AVAILABLE = False

class ClientDevice:
    def __init__(self, client_id, model: Any, context=None):
        """Initialize client device.
        
        Args:
            client_id: Unique identifier for this client
            model: Machine learning model (TensorFlowModel if TensorFlow is available)
            context: Encryption context (optional)
        """
        if not TENSORFLOW_AVAILABLE and isinstance(model, TensorFlowModel):
            raise RuntimeError("TensorFlow is not available")
            
        self.client_id = client_id
        self.model = model
        self.context = context
        self.connection = None
        self.logger = logging.getLogger(__name__)

    async def connect_to_central_server(self, uri):
        """Connect to central server and perform initial model architecture validation."""
        try:
            self.connection = await websockets.connect(uri)
            
            # Send initial model architecture for compatibility check
            model_arch = self.model.get_model_architecture()
            await self.send_message({
                'client_id': self.client_id,
                'model_architecture': model_arch
            })

            # Wait for validation response
            response = await self.receive_message()
            if response and 'error' in response:
                self.logger.error(f"Client {self.client_id}: {response['error']}")
                await self.connection.close()
                return False

            self.logger.info(f"Client {self.client_id}: Connected to central server at {uri}")
            return True
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error connecting to central server: {e}")
            return False

    async def send_message(self, message):
        try:
            serialized_message = pickle.dumps(message)
            await self.connection.send(serialized_message)
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error sending message: {e}")

    async def receive_message(self):
        try:
            message = await self.connection.recv()
            deserialized_message = pickle.loads(message)
            return deserialized_message
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error receiving message: {e}")
            return None

    async def federated_learning(self, x_train, y_train):
        """Participate in federated learning with model compatibility checks."""
        try:
            while True:
                message = await self.receive_message()
                if message is None:
                    break

                # Handle server messages
                if 'error' in message:
                    self.logger.error(f"Server error: {message['error']}")
                    break

                weights = message.get('weights', None)
                if weights is None:
                    break

                # Validate received weights
                decrypted_weights = decrypt_weights(self.context, weights)
                if not self.model.validate_weights(decrypted_weights):
                    self.logger.error("Received incompatible weights from server")
                    break

                # Update model and train
                self.model.set_weights(decrypted_weights)
                metrics = self.model.train(x_train, y_train, epochs=1)
                
                # Send updated weights and training metrics
                new_weights = self.model.get_weights()
                await self.send_message({
                    'weights': encrypt_weights(self.context, new_weights),
                    'num_samples': len(x_train),
                    'metrics': metrics
                })
                
                self.logger.info(f"Client {self.client_id}: Completed training round with metrics: {metrics}")
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error in federated learning loop: {e}")
            return False
        return True

    async def request_data(self):
        try:
            await self.send_message({'data_request': True})
            message = await self.receive_message()
            if message:
                data = message.get('data', None)
                self.logger.info(f"Client {self.client_id}: Received data from central server")
                return data
            return None
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error requesting data: {e}")
            return None
