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

batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]
logging.debug(example_batch)

# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  logging.debug(feature_layer(example_batch).numpy())

age = feature_column.numeric_column("age")
demo(age)

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# Notice the input to the embedding column is the categorical column
# we previously created
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))
# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)