import numpy as np
from federated_learning_framework.encryption import create_context, encrypt_weights, decrypt_weights

def test_encryption():
    context = create_context()
    weights = [np.random.rand(10, 10)]
    encrypted = encrypt_weights(context, weights)
    decrypted = decrypt_weights(context, encrypted)
    assert np.allclose(weights[0], decrypted[0])
