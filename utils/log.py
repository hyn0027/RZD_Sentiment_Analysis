import logging

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
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger