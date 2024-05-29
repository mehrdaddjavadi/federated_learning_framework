import logging
from federated_learning_framework.utils import setup_logging

def test_setup_logging():
    logger = setup_logging()
    
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO

    logger.info("Test logging setup")
