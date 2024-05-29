import asyncio
import pytest
from federated_learning_framework.central_server import CentralServer
from federated_learning_framework.encryption import create_context

@pytest.mark.asyncio
async def test_central_server():
    context = create_context()
    server = CentralServer(context=context)
    server_task = asyncio.create_task(server.run_server('localhost', 8089))

    await asyncio.sleep(1)  # Give the server some time to start

    # Simulate client connections and other test scenarios here

    server_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await server_task
