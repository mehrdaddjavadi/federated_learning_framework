import asyncio
import pytest
import tensorflow as tf
from federated_learning_framework.client_device import ClientDevice
from federated_learning_framework.encryption import create_context
from federated_learning_framework.models.tensorflow_model import TensorFlowModel

@pytest.mark.asyncio
async def test_client_device():
    context = create_context()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(3072,), activation='softmax')
    ])
    model = TensorFlowModel(model)
    client = ClientDevice(client_id=1, model=model, context=context)
    
    connect_task = asyncio.create_task(client.connect_to_central_server('ws://localhost:8089'))

    await asyncio.sleep(1)  # Give the client some time to connect

    # Simulate communication and other test scenarios here

    connect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connect_task
