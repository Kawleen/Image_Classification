import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import h5py
import pandas as pd
import numpy as np

class DataSet(object):

  def __init__(self, train_images, train_labels, train_class):
    self._num_examples = train_images.shape[0]
    
    self.train_images = train_images
    self.train_labels = train_labels
    self.train_class = train_class
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self.train_images

  @property
  def labels(self):
    return self.train_labels

  @property
  def img_names(self):
    return self.train_class

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self.train_images[start:end], self.train_labels[start:end], self.train_class[start:end]

def read_train_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  train_images = []
  train_labels = []
  train = pd.read_csv('/home/kawaleenm/Downloads/fashion_data/fashion-mnist_train.csv',skiprows = 0,low_memory=False)
  for rows in train.iterrows():
    index,data = rows
    img = data[1:].tolist()
    img = np.asarray(img,dtype = np.float32).reshape(28,28)
    train_images.append(img)
    
    index = data[0:1][0]
    label = np.zeros(10)
    label[index] = 1.0
    train_labels.append(label)

  test_images = []
  test_labels = []
  test = pd.read_csv('/home/kawaleenm/Downloads/fashion_data/fashion-mnist_test.csv',skiprows = 0,low_memory=False)
  for rows in test.iterrows():
    index,data = rows
    img = data[1:].tolist()
    img = np.asarray(img,dtype = np.float32).reshape(28,28)
    test_images.append(img)
    
    index = data[0:1][0]
    label = np.zeros(10)
    label[index] = 1.0
    test_labels.append(label)



  # test_image = test_images.reshape(60000,28,28,1)
  validation_images = np.asarray(test_images, dtype=np.float32)
  validation_images = validation_images.reshape(10000,28,28,1)
  validation_labels = np.asarray(test_labels, dtype=np.float32)
  validation_cls = ['0','1','2','3','4','5','6','7','8','9']


  # train_image = train_images.reshape(10000,28,28,1)
  train_images = np.asarray(train_images, dtype=np.float32)
  train_images = train_images.reshape(60000,28,28,1)
  train_labels = np.asarray(train_labels, dtype=np.float32)
  train_cls = ['0','1','2','3','4','5','6','7','8','9']

  data_sets.train = DataSet(train_images, train_labels,train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_cls)

  return data_sets