import logging
import os 


def get_logger(name:str):
    # Create Log Directory if doesn't exist
    log_dir =' Logs'
    os.makedirs(log_dir, exist_ok = True)


    # Create Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger


    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setLevel(logging.INFO)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S)')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger