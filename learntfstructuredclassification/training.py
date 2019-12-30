# all code from https://www.tensorflow.org/tutorials/structured_data/feature_columns
# I am following this tutorial to learn on my own!

from __future__ import absolute_import, division, print_function, unicode_literals


#DEBUG SET
import logging
logging.basicConfig(level=logging.DEBUG)


#IMPORT LIBRARIES
logging.debug("starting imports")
logging.debug("finished import absolute_import, division, print_function, unicode_literals...")
import numpy as np
logging.debug("finished import numpy as np...")
import pandas as pd
logging.debug("finished import pandas as pd...")

import tensorflow as tf
logging.debug("finished import tensorflow as tf...")
from tensorflow import feature_column
logging.debug("finished import feature_column")
from tensorflow.keras import layers
logging.debug("finished import layers")
from sklearn.model_selection import train_test_split
logging.debug("finished import layers\n")


#DATA IMPORT
logging.debug("started data download...")
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
logging.debug("finished data download...")

logging.debug(dataframe.head())
