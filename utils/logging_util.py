import logging
import sys


def start_logging():
    logging.basicConfig(level='INFO', format='%(message)s', stream=sys.stdout)
