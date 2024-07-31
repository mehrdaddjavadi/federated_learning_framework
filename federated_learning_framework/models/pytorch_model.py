import torch
import torch.nn as nn
from federated_learning_framework.models.abstract_model import AbstractModel

class PyTorchModel(AbstractModel):
    def __init__(self, model):
        self.model = model

    def get_weights(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_weights(self, weights):
        for param, weight in zip(self.model.parameters(), weights):
            param.data = torch.tensor(weight, dtype=param.data.dtype)

    def train(self, x_train, y_train, epochs=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    def predict(self, data):
        with torch.no_grad():
            return self.model(data).numpy()
