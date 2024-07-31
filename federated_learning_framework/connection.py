import asyncio
import websockets
import pickle
from websockets.exceptions import ConnectionClosedError

class ConnectionServer:
    def __init__(self, connection_type, host, port, client_handler):
        self.connection_type = connection_type
        self.host = host
        self.port = port
        self.client_handler = client_handler
        self.clients = {}

    async def start(self):
        if self.connection_type == 'websocket':
            async with websockets.serve(self.handle_client, self.host, self.port):
                await asyncio.Future()  # Run forever
        else:
            raise NotImplementedError(f"Connection type {self.connection_type} not supported")

    async def handle_client(self, websocket, path):
        client_id = len(self.clients) + 1
        self.clients[client_id] = websocket
        await self.client_handler(websocket, client_id)

    async def send(self, client_id, message):
        client = self.clients[client_id]
        serialized_message = pickle.dumps(message)
        await client.send(serialized_message)

    async def receive(self, client_id):
        client = self.clients[client_id]
        message = await client.recv()
        return pickle.loads(message)

class ConnectionClient:
    def __init__(self, connection_type, uri):
        self.connection_type = connection_type
        self.uri = uri
        self.connection = None

    async def connect(self):
        if self.connection_type == 'websocket':
            self.connection = await websockets.connect(self.uri)
        else:
            raise NotImplementedError(f"Connection type {self.connection_type} not supported")

    async def send(self, message):
        serialized_message = pickle.dumps(message)
        await self.connection.send(serialized_message)

    async def receive(self):
        message = await self.connection.recv()
        return pickle.loads(message)
