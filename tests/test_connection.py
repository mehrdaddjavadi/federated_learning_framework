import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from federated_learning_framework.connection import (
    ConnectionServer,
    ConnectionClient
)

@pytest.fixture
def mock_websocket():
    websocket = AsyncMock()
    websocket.send = AsyncMock()
    websocket.recv = AsyncMock()
    websocket.close = AsyncMock()
    return websocket

@pytest.mark.asyncio
async def test_server_initialization():
    handler = AsyncMock()
    server = ConnectionServer('websocket', 'localhost', 8765, handler)
    assert server.connection_type == 'websocket'
    assert server.host == 'localhost'
    assert server.port == 8765
    assert server.client_handler == handler

@pytest.mark.asyncio
async def test_server_client_handling(mock_websocket):
    handler = AsyncMock()
    server = ConnectionServer('websocket', 'localhost', 8765, handler)
    
    # Test client connection handling
    with patch('websockets.serve', new_callable=AsyncMock) as mock_serve:
        # Start server in background task
        server_task = asyncio.create_task(server.start())
        await asyncio.sleep(0.1)
        
        # Simulate client connection
        await server.handle_client(mock_websocket, '/test')
        
        # Check client was added
        client_id = list(server.connection_manager.get_active_clients())[0]
        assert client_id
        
        # Check client status
        status = server.get_client_status(client_id)
        assert status['state'] == 'connected'
        
        # Cleanup
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

@pytest.mark.asyncio
async def test_server_message_sending(mock_websocket):
    handler = AsyncMock()
    server = ConnectionServer('websocket', 'localhost', 8765, handler)
    
    # Simulate client connection
    await server.handle_client(mock_websocket, '/test')
    client_id = list(server.connection_manager.get_active_clients())[0]
    
    # Test sending message
    test_message = {'type': 'test', 'data': 'hello'}
    success = await server.send(client_id, test_message)
    assert success
    mock_websocket.send.assert_awaited_once()

@pytest.mark.asyncio
async def test_server_message_broadcasting(mock_websocket):
    handler = AsyncMock()
    server = ConnectionServer('websocket', 'localhost', 8765, handler)
    
    # Simulate multiple client connections
    mock_websockets = [AsyncMock(), AsyncMock(), AsyncMock()]
    client_ids = []
    
    for websocket in mock_websockets:
        await server.handle_client(websocket, '/test')
        client_ids.append(list(server.connection_manager.get_active_clients())[-1])
    
    # Test broadcasting
    test_message = {'type': 'broadcast', 'data': 'hello all'}
    results = await server.broadcast(test_message)
    
    assert len(results) == 3
    assert all(results.values())
    for websocket in mock_websockets:
        websocket.send.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_initialization():
    client = ConnectionClient('websocket', 'ws://localhost:8765')
    assert client.connection_type == 'websocket'
    assert client.uri == 'ws://localhost:8765'
    assert client.connection is None

@pytest.mark.asyncio
async def test_client_connection():
    client = ConnectionClient('websocket', 'ws://localhost:8765')
    
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        # Test connection
        success = await client.connect()
        assert success
        assert client.connection == mock_websocket
        mock_connect.assert_awaited_once_with('ws://localhost:8765')

@pytest.mark.asyncio
async def test_client_message_sending():
    client = ConnectionClient('websocket', 'ws://localhost:8765')
    
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        # Connect client
        await client.connect()
        
        # Test sending message
        test_message = {'type': 'test', 'data': 'hello'}
        success = await client.send(test_message)
        assert success
        mock_websocket.send.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_message_receiving():
    client = ConnectionClient('websocket', 'ws://localhost:8765')
    
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        # Setup mock response
        test_message = {'type': 'response', 'data': 'hello back'}
        mock_websocket.recv.return_value = test_message
        
        # Connect client
        await client.connect()
        
        # Test receiving message
        received = await client.receive()
        assert received == test_message
        mock_websocket.recv.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_auto_reconnection():
    client = ConnectionClient('websocket', 'ws://localhost:8765')
    
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        # Connect client
        await client.connect()
        
        # Simulate connection loss
        mock_websocket.wait_closed.return_value = True
        mock_websocket.closed = True
        
        # Try sending message to trigger reconnection
        await client.send({'type': 'test'})
        
        # Verify reconnection attempt
        assert mock_connect.call_count > 1

@pytest.mark.asyncio
async def test_client_graceful_shutdown():
    client = ConnectionClient('websocket', 'ws://localhost:8765')
    
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        # Connect client
        await client.connect()
        
        # Close client
        await client.close()
        
        # Verify cleanup
        mock_websocket.close.assert_awaited_once()
        assert client.connection is None
        assert not client._connected.is_set()
