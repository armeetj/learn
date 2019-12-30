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
#read csv into dataframe object
dataframe = pd.read_csv(URL)
logging.debug("finished data download...\n")

#DATA SPLITTING
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
logging.debug(str(len(train)) + ' train examples')
logging.debug(str(len(val)) + ' validation examples')
logging.debug(str(len(test)) + ' test examples\n')

#Convert dataframe to tf.data dataset
#A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds