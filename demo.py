from CLV_Prediction.exception import CLVexception
from CLV_Prediction.logger import logging
import sys


try:
    a = 2/'a'
except Exception as a:
    logging.debug(a)
    raise CLVexception(a, sys) from a
