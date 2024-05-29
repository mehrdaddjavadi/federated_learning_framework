import asyncio
import pytest
import tensorflow as tf
from federated_learning_framework.client_device import ClientDevice

@pytest.mark.asyncio
async def test_client_device():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(3072,), activation='softmax')
    ])
    client = ClientDevice(1, model, None)
    await client.connect_to_central_server()
    # Other optinal test cases
