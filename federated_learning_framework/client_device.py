import asyncio
import logging
import websockets
from federated_learning_framework.encryption import encrypt_weights, decrypt_weights
from federated_learning_framework.models.abstract_model import AbstractModel

class ClientDevice:
    def __init__(self, client_id, model: AbstractModel, context):
        self.client_id = client_id
        self.model = model
        self.context = context
        self.connection = None
        self.logger = logging.getLogger(__name__)

    async def connect_to_central_server(self, uri):
        try:
            self.connection = await websockets.connect(uri)
            await self.connection.send(str(self.client_id))
            self.logger.info(f"Client {self.client_id}: Connected to central server at {uri}")
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error connecting to central server: {e}")

    async def federated_learning(self, x_train, y_train):
        try:
            while True:
                weights = await self.receive_weights()
                self.model.set_weights(decrypt_weights(self.context, weights))
                self.model.train(x_train, y_train, epochs=1)
                new_weights = self.model.get_weights()
                await self.send_weights(encrypt_weights(self.context, new_weights))
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error in federated learning loop: {e}")

    async def receive_weights(self):
        try:
            message = await self.connection.recv()
            self.logger.info(f"Client {self.client_id}: Received weights")
            return message['weights']
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error receiving weights: {e}")

    async def send_weights(self, weights):
        try:
            await self.connection.send({'weights': weights})
            self.logger.info(f"Client {self.client_id}: Sent weights to central server")
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error sending weights: {e}")

    async def request_data(self):
        try:
            await self.connection.send({'data_request': True})
            data = await self.connection.recv()
            self.logger.info(f"Client {self.client_id}: Received data from central server")
            return data['data']
        except Exception as e:
            self.logger.error(f"Client {self.client_id}: Error requesting data: {e}")
