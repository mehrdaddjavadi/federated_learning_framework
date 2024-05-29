import numpy as np
import tensorflow as tf
from federated_learning_framework.active_learning import query_active_learning

def test_active_learning():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(3072,), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    unlabeled_data = np.random.rand(100, 3072)
    selected_indices = query_active_learning(model, unlabeled_data, 5)
    assert len(selected_indices) == 5
