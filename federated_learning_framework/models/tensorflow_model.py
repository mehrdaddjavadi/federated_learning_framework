import tensorflow as tf
from federated_learning_framework.models.abstract_model import AbstractModel

class TensorFlowModel(AbstractModel):
    def __init__(self, model):
        self.model = model

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train(self, x_train, y_train, epochs=1):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=0)

    def predict(self, data):
        return self.model.predict(data)
