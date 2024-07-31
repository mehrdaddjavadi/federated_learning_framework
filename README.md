# Federated Learning Framework

## Overview

Welcome to the Federated Learning Framework, a modular and extensible solution for implementing federated learning across various applications. Harness the power of collective intelligence, ensure data privacy with homomorphic encryption, and apply it to domains like NLP, autonomous vehicles, drones, and more.

## Features

- **Modular and Extensible**: Easily customizable for different machine learning and deep learning applications.
- **Secure**: Utilizes homomorphic encryption to ensure data privacy.
- **Active Learning**: Incorporates active learning strategies to improve model performance.
- **Flexible Communication**: Supports various connection methods including socket programming.
- **Customizable**: Users can edit and control every part of the framework with various functions.

## Potential Applications

### Healthcare

Federated learning can be used to train models on patient data from multiple hospitals without sharing sensitive information. This approach can improve medical diagnostics and treatment recommendations while preserving patient privacy.

### Autonomous Vehicles

By collecting and learning from data across multiple autonomous vehicles, the framework can help improve the safety and performance of self-driving cars without exposing individual vehicle data.

### Drones

Drones can use federated learning to share and learn from data collected during their operations, enhancing their navigation, object detection, and other capabilities while ensuring data security.

### Natural Language Processing (NLP)

Federated learning can be applied to train NLP models on data from multiple sources, such as user devices, to improve language understanding and generation without compromising user privacy.

### Finance

Financial institutions can use federated learning to develop fraud detection and risk management models by leveraging data from multiple sources while keeping customer data secure.

### Smart Homes and IoT Devices

IoT devices in smart homes can collaboratively learn from user interactions to optimize performance and provide better services without sharing raw data.

## Detailed Component Description

### Central Server

**File:** `central_server.py`

The central server orchestrates the federated learning process by coordinating the communication and aggregation of model weights from various client devices.

**Key Functions:**

- `run_server`: Starts the server to handle client connections.
- `handle_client`: Manages incoming messages from clients.
- `transmit_weights`: Broadcasts the aggregated weights to clients.
- `send_data_to_client`: Sends specific data to a client.
- `get_data_from_client`: Requests and receives data from a client.
- `query_active_learning`: Implements active learning strategies to select data for labeling.

### Client Device

**File:** `client_device.py`

Client devices perform local training on their datasets and communicate with the central server.

**Key Functions:**

- `connect_to_central_server`: Connects to the central server.
- `federated_learning`: Coordinates local training and communication with the server.
- `receive_weights`: Receives model weights from the central server.
- `send_weights`: Sends model weights to the central server.
- `receive_data`: Receives data from the central server.

### Encryption

**File:** `encryption.py`

Provides functions for creating encryption contexts and encrypting/decrypting model weights.

**Key Functions:**

- `create_context`: Sets up the encryption context using TenSEAL.
- `encrypt_weights`: Encrypts model weights.
- `decrypt_weights`: Decrypts encrypted model weights.

### Active Learning

**File:** `active_learning.py`

Implements active learning strategies to enhance the training process by selectively querying informative data points.

**Key Functions:**

- `select_informative_samples`: Selects samples for labeling based on uncertainty.

### Connection

**File:** `connection.py`

Manages the connection types and protocols (e.g., WebSocket) for communication between the central server and client devices.

**Key Functions:**

- `run_server`: Starts a WebSocket server.
- `connect_to_server`: Establishes a WebSocket connection to the server.

### Decorators

**File:** `decorators.py`

Provides decorators for adding federated learning and encryption functionalities to functions.

**Key Functions:**

- `federated_learning_decorator`: Wraps a function to enable federated learning.
- `encryption_decorator`: Wraps a function to enable homomorphic encryption.

### Utilities

**File:** `utils.py`

Includes utility functions used throughout the framework.

## Installation

Clone the repository:

```sh
git clone https://github.com/mehrdaddjavadi/federated_learning_framework.git
```

Navigate to the directory:

```sh
cd federated_learning_framework
```

Install the dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Setting Up the Central Server

```python
import asyncio
from federated_learning_framework.central_server import CentralServer

async def main():
    server = CentralServer()
    await server.run_server()

asyncio.run(main())
```

### Setting Up the Central Server On Interactive Environment Like Jupyter Notebook

```python
import nest_asyncio
import asyncio
from federated_learning_framework.central_server import CentralServer

nest_asyncio.apply()

async def main():
    server = CentralServer()
    await server.run_server()

# If running in an environment with an existing event loop
if __name__ == "__main__":
    asyncio.run(main())
```

### Setting Up a Client Device

```python
import asyncio
import tensorflow as tf
from federated_learning_framework.client_device import ClientDevice
from federated_learning_framework.encryption import create_context

# Define your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3072,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Create context for encryption
context = create_context()

# Initialize the client device
client = ClientDevice(client_id=1, model=model, context=context)

async def main():
    uri = "ws://localhost:8089"
    await client.connect_to_central_server(uri)
    x_train, y_train = ...  # Load your training data
    await client.federated_learning(uri, x_train, y_train)
    # Optionally receive data from central server
    data = await client.receive_data()
    print(f"Received data: {data}")

asyncio.run(main())
```

### Sample Execution Script Using Decorators For Interactive Environments Like Colab And Jupyter Notebook

```python

import asyncio
import nest_asyncio
import tensorflow as tf
from federated_learning_framework.central_server import CentralServer
from federated_learning_framework.client_device import ClientDevice
from federated_learning_framework.encryption import create_context
from federated_learning_framework.models.tensorflow_model import TensorFlowModel

nest_asyncio.apply()

async def run_federated_learning():
    # Setup models
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(3072,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    tf_model = TensorFlowModel(model)

    # Create encryption context
    context = create_context()

    # Initialize server and clients
    central_server = CentralServer(connection_type='websocket', context=context)
    client1 = ClientDevice(client_id=1, model=tf_model, context=context)
    client2 = ClientDevice(client_id=2, model=tf_model, context=context)

    # Define URIs for the server
    uri = "ws://localhost:8089"

    # Run the server
    async def server_task():
        await central_server.run_server()

    # Connect clients and run federated learning
    async def client_task():
        await client1.connect_to_central_server(uri)
        await client2.connect_to_central_server(uri)

        x_train = tf.random.normal((10, 3072))
        y_train = tf.random.uniform((10,), maxval=10, dtype=tf.int32)

        await asyncio.gather(
            client1.federated_learning(x_train, y_train),
            client2.federated_learning(x_train, y_train)
        )

    # Run both server and client tasks
    await asyncio.gather(
        server_task(),
        client_task()
    )

# Execute the main federated learning function
await run_federated_learning()


```

### Sample 2 Execution Script Using Decorators For Interactive Environments Like Colab And Jupyter Notebook

```python
import asyncio
import tensorflow as tf
import numpy as np
from federated_learning_framework.client_device import ClientDevice
from federated_learning_framework.central_server import CentralServer
from federated_learning_framework.encryption import create_context
from federated_learning_framework.models.tensorflow_model import TensorFlowModel

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3072,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
wrapped_model = TensorFlowModel(model)

# Create encryption context
context = create_context()

# Initialize server and clients
central_server = CentralServer(context=context)
client1 = ClientDevice(client_id=1, model=wrapped_model, context=context)
client2 = ClientDevice(client_id=2, model=wrapped_model, context=context)

# Dummy training data
x_train = np.random.rand(10, 3072)
y_train = np.random.randint(0, 10, 10)

async def main():
    await asyncio.gather(
        central_server.run_server(),
        client1.connect_to_central_server("ws://localhost:8089"),
        client2.connect_to_central_server("ws://localhost:8089"),
        client1.federated_learning(x_train, y_train),
        client2.federated_learning(x_train, y_train)
    )

asyncio.run(main())

```

### Using Decorators

```python
import asyncio
import tensorflow as tf
from federated_learning_framework.decorators import federated_learning_decorator, encryption_decorator
from federated_learning_framework.client_device import ClientDevice
from federated_learning_framework.encryption import create_context

# Create context for encryption
context = create_context()

# Define your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3072,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

@federated_learning_decorator(uri="ws://localhost:8089")
@encryption_decorator(context=context)
async def main():
    client = ClientDevice(client_id=1, model=model, context=context)
    await client.connect_to_central_server('ws://localhost:8089')
    x_train, y_train = ...  # Load your training data
    await client.federated_learning('ws://localhost:8089', x_train, y_train)

asyncio.run(main())
```

## Running Tests

To run the tests, execute the following command in the root directory:

```sh
python -m unittest discover -s tests
```

## License

The usage of this library is free for academic work with proper referencing. For business, governmental, and any other types of usage, please contact me directly. All rights are reserved.

**Contact:** mehrdaddjavadi@gmail.com

## Contributing

Feel free to contribute by submitting a pull request or opening an issue.

```

Copy and paste this into your README.md file. This format provides a clear, organized structure and includes all necessary details and instructions for potential users and contributors.
```
