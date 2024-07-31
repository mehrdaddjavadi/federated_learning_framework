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
import tensorflow as tf
from federated_learning_framework.encryption import encrypt_weights, decrypt_weights
from federated_learning_framework.models.tensorflow_model import TensorFlowModel
import websockets
import pickle

class ClientDevice:
    def __init__(self, client_id, model: TensorFlowModel, context):
        self.client_id = client_id
        self.model = model
        self.context = context
        self.connection = None
        self.logger = logging.getLogger(__name__)

    async def connect_to_central_server(self, uri):
        try:
            self.connection = await websockets.connect(uri)
            await self.send_message({'client_id': self.client_id})
            self.logger.info(f"Client {self.client_id}: Connected to central server at {uri}")
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error connecting to central server: {e}")

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
        try:
            while True:
                message = await self.receive_message()
                if message is None:
                    break
                weights = message.get('weights', None)
                if weights is None:
                    break

                self.model.set_weights(decrypt_weights(self.context, weights))
                self.model.train(x_train, y_train, epochs=1)
                new_weights = self.model.get_weights()
                await self.send_message({'weights': encrypt_weights(self.context, new_weights)})
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error in federated learning loop: {e}")

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
