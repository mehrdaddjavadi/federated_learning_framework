from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    @abstractmethod
    def train(self, x_train, y_train, epochs=1):
        pass

    @abstractmethod
    def predict(self, data):
        pass
