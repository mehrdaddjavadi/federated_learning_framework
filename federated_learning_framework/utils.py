import logging

def setup_logging(logfile='federated_learning.log'):
    """Set up logging configuration and return a logger instance.
    
    Args:
        logfile: Path to log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger
