import asyncio
import logging
import numpy as np
from federated_learning_framework.connection import ConnectionServer
from websockets.exceptions import ConnectionClosedError
from federated_learning_framework.encryption import create_context

class CentralServer:
    def __init__(self, connection_type='websocket', host='0.0.0.0', port=8089, context=None):
        self.model_weights = None
        self.lock = asyncio.Lock()
        self.clients = set()
        self.logger = logging.getLogger(__name__)
        self.connection = ConnectionServer(connection_type, host, port, self.handle_client)
        self.context = context or create_context()

    async def run_server(self):
        self.logger.info("Central Server is starting...")
        await self.connection.start()

    async def handle_client(self, connection, client_id):
        self.clients.add(connection)
        self.logger.info(f"Central Server: Client {client_id} connected")
        try:
            while True:
                message = await connection.receive()
                await self.transmit_weights(message)
                # Optionally send data to the client
                await self.send_data_to_client(client_id, {'data': 'sample data'})
        except ConnectionClosedError:
            self.logger.info(f"Central Server: Client {client_id} disconnected")
        finally:
            self.clients.remove(connection)

    async def transmit_weights(self, weights):
        async with self.lock:
            self.model_weights = weights
            await asyncio.gather(*[client.send(self.model_weights) for client in self.clients])
            self.logger.info("Central Server: Received weights from clients")

    async def send_data_to_client(self, client_id, data):
        self.logger.info(f"Central Server: Sending data to client {client_id}")
        await asyncio.gather(*[client.send(data) for client in self.clients if client.id == client_id])

    async def get_data_from_client(self, client_id):
        self.logger.info(f"Central Server: Requesting data from client {client_id}. Simulating response.")
        await asyncio.sleep(1)
        return np.random.rand(10, 3072)

    def query_active_learning(self, unlabeled_data, model):
        uncertainty = model.predict(unlabeled_data)
        selected_indices = np.argsort(uncertainty.max(axis=1))[:5]
        return selected_indices
