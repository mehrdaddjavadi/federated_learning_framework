import asyncio
import numpy as np
import tensorflow as tf
from federated_learning_framework.connection import ConnectionClient
from websockets.exceptions import ConnectionClosedError, InvalidURI, WebSocketException

class ClientDevice:
    def __init__(self, client_id, model, context, connection_type='websocket', uri='ws://localhost:8089'):
        self.client_id = client_id
        self.model = model
        self.context = context
        self.connection = ConnectionClient(connection_type, uri)
        self.toggle_encryption = True

    async def connect_to_central_server(self):
        try:
            await self.connection.connect()
            await self.connection.send(f"Client {self.client_id}: Connected")
            while True:
                message = await self.connection.receive()
                print(f"Client {self.client_id}: Received message - {message}")
        except ConnectionClosedError as e:
            print(f"Client {self.client_id}: Connection to the central server closed unexpectedly - {e}")
        except InvalidURI as e:
            print(f"Client {self.client_id}: Invalid URI for the central server - {e}")
        except WebSocketException as e:
            print(f"Client {self.client_id}: WebSocket error - {e}")
        except Exception as e:
            print(f"Client {self.client_id}: An unexpected error occurred - {e}")

    async def federated_learning(self, central_server, x_train, y_train):
        for _ in range(10):
            self.train_model(x_train, y_train)

            if self.toggle_encryption:
                encrypted_weights = self.encrypt_model_weights()
                await central_server.transmit_weights(encrypted_weights)
                print(f"Client {self.client_id}: Transmitted encrypted weights to the Central Server")

                print(f"Client {self.client_id}: Requesting unlabeled data from the Central Server...")
                unlabeled_data = await central_server.get_data_from_client(self.client_id)

                print(f"Client {self.client_id}: Querying active learning strategy...")
                selected_indices = central_server.query_active_learning(unlabeled_data, self.model)

                labels = np.random.randint(0, 10, size=len(selected_indices))
                print(f"Client {self.client_id}: Updating local model with labeled instances...")
                self.model.fit(unlabeled_data[selected_indices], labels, epochs=1, batch_size=32, verbose=0)

    def train_model(self, data, labels):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(data, labels, epochs=1, batch_size=32, verbose=0)

    def encrypt_model_weights(self):
        encrypted_weights = []
        for weight in self.model.get_weights():
            weight_array = np.array(weight)
            flat_weight = weight_array.flatten()
            flat_weight = flat_weight.reshape((-1,))
            flat_weight = flat_weight.tolist()
            encrypted_vector = ts.ckks_vector(self.context, flat_weight)
            encrypted_weights.append(encrypted_vector.serialize())
        return encrypted_weights
