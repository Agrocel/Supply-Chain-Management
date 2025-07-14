import logging
import os

def get_logger(name: str):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs during multiple imports/runs
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    #logger.addHandler(console_handler)

    return logger
