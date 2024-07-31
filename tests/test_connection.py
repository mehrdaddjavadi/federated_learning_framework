import asyncio
import pytest
from federated_learning_framework.connection import ConnectionServer, ConnectionClient

@pytest.mark.asyncio
async def test_connection():
    async def handle_client(connection, client_id):
        message = await connection.receive()
        await connection.send(message)

    server = ConnectionServer('websocket', 'localhost', 8089, handle_client)
    server_task = asyncio.create_task(server.start())

    await asyncio.sleep(1)  # Give the server some time to start

    client = ConnectionClient('websocket', 'ws://localhost:8089')
    await client.connect()

    test_message = "test_message"
    await client.send(test_message)
    received_message = await client.receive()

    assert received_message == test_message

    await client.connection.close()
    server_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await server_task
