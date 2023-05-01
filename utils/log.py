import logging
import tqdm

def getLogger(args, name, output=True):
    logger = logging.getLogger(name)
    if type(args) == str:
        logger.setLevel(eval("logging." + args))
    else:
        logger.setLevel(eval("logging." + args.verbose))
    if output:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  