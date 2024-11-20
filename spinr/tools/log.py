import logging
import sys

###
# Project wide logging settings
###

log_level = logging.INFO
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

def terminal_log(log=None):

    if log is None:
        log = create_terminal_log()
    elif log is not None:
        add_terminal_log(log)

    return log

def file_log(log_file, log=None):

    if log is None:
        log = create_file_log(log_file)
    elif log is not None:
        add_file_log(log, log_file=log_file)

    return log

def create_terminal_log():
    log = logging.getLogger("default")
    term_handler = logging.StreamHandler(sys.stdout)
    term_handler.setFormatter(formatter)

    log.setLevel(log_level)
    log.addHandler(term_handler)

    return log

def create_file_log(log_file):
    log = logging.getLogger("default")
    handler = logging.FileHandler(log_file, mode='w')        
    handler.setFormatter(formatter)

    log.setLevel(log_level)
    log.addHandler(handler)

    return log

def add_terminal_log(log):
    term_handler = logging.StreamHandler(sys.stdout)
    term_handler.setFormatter(formatter)

    log.setLevel(log_level)
    log.addHandler(term_handler)

    return log

def add_file_log(log, log_file):
    handler = logging.FileHandler(log_file, mode='w')        
    handler.setFormatter(formatter)

    log.setLevel(log_level)
    log.addHandler(handler)

    return log
