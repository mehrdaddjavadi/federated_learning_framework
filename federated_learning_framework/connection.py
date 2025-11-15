# import asyncio
# import websockets
# import pickle
# from websockets.exceptions import ConnectionClosedError

# class ConnectionServer:
#     def __init__(self, connection_type, host, port, client_handler):
#         self.connection_type = connection_type
#         self.host = host
#         self.port = port
#         self.client_handler = client_handler
#         self.clients = {}

#     async def start(self):
#         if self.connection_type == 'websocket':
#             async with websockets.serve(self.handle_client, self.host, self.port):
#                 await asyncio.Future()  # Run forever
#         else:
#             raise NotImplementedError(f"Connection type {self.connection_type} not supported")

#     async def handle_client(self, websocket, path):
#         client_id = len(self.clients) + 1
#         self.clients[client_id] = websocket
#         await self.client_handler(websocket, client_id)

#     async def send(self, client_id, message):
#         client = self.clients[client_id]
#         serialized_message = pickle.dumps(message)
#         await client.send(serialized_message)

#     async def receive(self, client_id):
#         client = self.clients[client_id]
#         message = await client.recv()
#         return pickle.loads(message)

# class ConnectionClient:
#     def __init__(self, connection_type, uri):
#         self.connection_type = connection_type
#         self.uri = uri
#         self.connection = None

#     async def connect(self):
#         if self.connection_type == 'websocket':
#             self.connection = await websockets.connect(self.uri)
#         else:
#             raise NotImplementedError(f"Connection type {self.connection_type} not supported")

#     async def send(self, message):
#         serialized_message = pickle.dumps(message)
#         await self.connection.send(serialized_message)

#     async def receive(self):
#         message = await self.connection.recv()
#         return pickle.loads(message)

import asyncio
import websockets
import pickle
import logging
from typing import Any, Callable, Dict, Optional
from websockets.exceptions import ConnectionClosedError
from .connection_manager import ConnectionManager

class ConnectionServer:
    def __init__(self, connection_type: str, host: str, port: int, client_handler: Callable):
        """Initialize connection server.
        
        Args:
            connection_type: Type of connection ('websocket' only for now)
            host: Host address to bind to
            port: Port number to listen on
            client_handler: Callback for handling client messages
        """
        self.connection_type = connection_type
        self.host = host
        self.port = port
        self.client_handler = client_handler
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager(
            max_reconnect_attempts=3,
            reconnect_timeout=30.0,
            heartbeat_interval=10.0,
            message_timeout=5.0
        )

    async def start(self):
        """Start the connection server."""
        if self.connection_type == 'websocket':
            async with websockets.serve(self.handle_client, self.host, self.port):
                self.logger.info(f"Server started on {self.host}:{self.port}")
                await asyncio.Future()  # Run forever
        else:
            raise NotImplementedError(f"Connection type {self.connection_type} not supported")

    async def handle_client(self, websocket: Any, path: str):
        """Handle new client connection."""
        # Generate unique client ID
        client_id = str(hash(websocket))
        self.logger.info(f"New client connected: {client_id}")
        
        # Handle connection through connection manager
        await self.connection_manager.handle_connection(
            websocket,
            client_id,
            self._handle_client_message
        )

    async def _handle_client_message(self, client_id: str, raw_message: bytes):
        """Handle incoming client message."""
        try:
            # Deserialize message
            message = pickle.loads(raw_message)
            
            # Pass to client handler
            await self.client_handler(client_id, message)
            
        except pickle.PickleError as e:
            self.logger.error(f"Failed to deserialize message from client {client_id}: {e}")
        except Exception as e:
            self.logger.error(f"Error handling message from client {client_id}: {e}")

    async def send(self, client_id: str, message: Any, require_ack: bool = True) -> bool:
        """Send message to specific client.
        
        Args:
            client_id: ID of client to send to
            message: Message to send
            require_ack: Whether to wait for acknowledgment
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            serialized_message = pickle.dumps(message)
            success = await self.connection_manager.send_message(
                client_id,
                serialized_message,
                require_ack
            )
            
            if success:
                self.logger.debug(f"Message sent to client {client_id}")
            else:
                self.logger.warning(f"Failed to send message to client {client_id}")
                
            return success
            
        except pickle.PickleError as e:
            self.logger.error(f"Failed to serialize message for client {client_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending message to client {client_id}: {e}")
            return False

    async def broadcast(self, 
                       message: Any,
                       clients: Optional[set] = None,
                       require_ack: bool = False) -> Dict[str, bool]:
        """Broadcast message to multiple clients.
        
        Args:
            message: Message to broadcast
            clients: Set of client IDs to send to (None for all)
            require_ack: Whether to wait for acknowledgments
            
        Returns:
            Dict[str, bool]: Map of client IDs to success status
        """
        try:
            serialized_message = pickle.dumps(message)
            results = await self.connection_manager.broadcast_message(
                serialized_message,
                clients,
                require_ack
            )
            
            # Log results
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            self.logger.info(
                f"Broadcast complete: {success_count}/{total_count} successful"
            )
            
            return results
            
        except pickle.PickleError as e:
            self.logger.error(f"Failed to serialize broadcast message: {e}")
            return {client_id: False for client_id in (clients or [])}
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
            return {client_id: False for client_id in (clients or [])}

    def get_active_clients(self) -> set:
        """Get set of currently active client IDs."""
        return self.connection_manager.get_active_clients()

    def get_client_status(self, client_id: str) -> Dict[str, Any]:
        """Get status information for specific client."""
        return self.connection_manager.get_client_status(client_id)


class ConnectionClient:
    def __init__(self, connection_type: str, uri: str):
        """Initialize connection client.
        
        Args:
            connection_type: Type of connection ('websocket' only for now)
            uri: URI to connect to
        """
        self.connection_type = connection_type
        self.uri = uri
        self.connection = None
        self.logger = logging.getLogger(__name__)
        self._reconnect_task = None
        self._connected = asyncio.Event()
        self._message_queue = asyncio.Queue()

    async def connect(self, max_retries: int = 3, retry_delay: float = 5.0):
        """Connect to server with retry logic."""
        if self.connection_type == 'websocket':
            for attempt in range(max_retries):
                try:
                    self.connection = await websockets.connect(self.uri)
                    self.logger.info(f"Connected to server at {self.uri}")
                    self._connected.set()
                    
                    # Start background tasks
                    self._reconnect_task = asyncio.create_task(
                        self._auto_reconnect()
                    )
                    return True
                    
                except Exception as e:
                    self.logger.error(
                        f"Connection attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        
            self.logger.error("Failed to connect after maximum retries")
            return False
            
        else:
            raise NotImplementedError(
                f"Connection type {self.connection_type} not supported"
            )

    async def send(self, message: Any, timeout: float = 5.0) -> bool:
        """Send message to server.
        
        Args:
            message: Message to send
            timeout: Timeout in seconds
            
        Returns:
            bool: True if message was sent successfully
        """
        if not await self._ensure_connected():
            return False
            
        try:
            serialized_message = pickle.dumps(message)
            await asyncio.wait_for(
                self.connection.send(serialized_message),
                timeout=timeout
            )
            self.logger.debug(f"Sent message: {message}")
            return True
            
        except asyncio.TimeoutError:
            self.logger.warning("Send operation timed out")
            return False
        except pickle.PickleError as e:
            self.logger.error(f"Failed to serialize message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            await self._handle_connection_error()
            return False

    async def receive(self, timeout: float = None) -> Optional[Any]:
        """Receive message from server.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Received message or None if error/timeout
        """
        if not await self._ensure_connected():
            return None
            
        try:
            raw_message = await asyncio.wait_for(
                self.connection.recv(),
                timeout=timeout
            )
            
            message = pickle.loads(raw_message)
            self.logger.debug(f"Received message: {message}")
            return message
            
        except asyncio.TimeoutError:
            self.logger.warning("Receive operation timed out")
            return None
        except pickle.PickleError as e:
            self.logger.error(f"Failed to deserialize message: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            await self._handle_connection_error()
            return None

    async def close(self):
        """Close connection gracefully."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            
        if self.connection:
            try:
                await self.connection.close()
                self.logger.info("Connection closed")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
                
        self._connected.clear()
        self.connection = None

    async def _ensure_connected(self) -> bool:
        """Ensure connection is active."""
        if not self.connection or self.connection.closed:
            self.logger.warning("Connection lost, attempting to reconnect")
            return await self.connect()
        return True

    async def _auto_reconnect(self):
        """Background task for automatic reconnection."""
        while True:
            try:
                await self.connection.wait_closed()
                self.logger.warning("Connection lost, initiating reconnect")
                self._connected.clear()
                
                # Attempt reconnection
                if await self.connect():
                    self.logger.info("Reconnection successful")
                else:
                    self.logger.error("Reconnection failed")
                    return
                    
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.logger.error(f"Error in reconnection task: {e}")
                await asyncio.sleep(5.0)

    async def _handle_connection_error(self):
        """Handle connection errors."""
        self._connected.clear()
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                pass
        self.connection = None
