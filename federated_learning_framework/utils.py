import logging

def setup_logging(logfile='federated_learning.log'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
