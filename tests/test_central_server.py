import asyncio
import pytest
from federated_learning_framework.central_server import CentralServer

@pytest.mark.asyncio
async def test_central_server():
    server = CentralServer()
    await server.run_server()
    # Other optional test cases
