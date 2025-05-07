import logging

def get_logger(name="pccl"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
