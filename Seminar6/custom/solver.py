################################################# You MIGHT need these imports.
import sys
from net import Net
from copy import deepcopy

from fast_rcnn.config import cfg
from roi_data_layer.layer import RoIDataLayer
from custom.utilities import *
import theano
import lasagne
import theano.tensor as T

class Solver(object):
  def __init__(self, roidb, net):
    # Holds current iteration number. 
    self.iter = 0

    # How frequently we should print the training info.
    self.display_freq = 1

    # Holds the path prefix for snapshots.
    self.snapshot_prefix = 'snapshot'
    
    self.roidb = roidb
    self.net = net
    self.roi_data_layer = RoIDataLayer()
    self.roi_data_layer.setup()
    self.roi_data_layer.set_roidb(self.roidb)
    self.stepfn = self.build_step_fn(self.net)

    ###################################################### Your code goes here.

  # This might be a useful static method to have.
  @staticmethod
  def build_step_fn(net):
    target_y = T.vector("target Y",dtype='int64')
    loss = lasagne.objectives.categorical_crossentropy(net.prediction,target_y).mean()
    accuracy = lasagne.objectives.categorical_accuracy(net.prediction,target_y).mean()
    updates_sgd = lasagne.updates.sgd(loss, net.params, learning_rate=0.0001)
    stepfn = theano.function([net.inp, target_y], [loss, accuracy], updates=updates_sgd, allow_input_downcast=True)
    return stepfn


  def get_training_batch(self):
    """Uses ROIDataLayer to fetch a training batch.

    Returns:
      input_data (ndarray): input data suitable for R-CNN processing
      labels (ndarray): batch labels (of type int32)
    """
    data, rois, labels = deepcopy(self.roi_data_layer.top[: 3])
    X = roi_pool(data, rois)
    y = labels.astype('int')
    
    return X, y

  def step(self):
    self.roi_data_layer.forward()
    data, labels = self.get_training_batch()
    """Conducts a single step of SGD."""
    
    loss, acc = self.stepfn(data, labels)
    
    self.loss = loss
    self.acc = acc
    ###################################################### Your code goes here.
    # Among other things, assign the current loss value to self.loss.

    self.iter += 1
    if self.iter % self.display_freq == 0:
      print 'Iteration {:<5} Train loss: {} Train acc: {}'.format(self.iter, self.loss, self.acc)

  def save(self, filename):
    self.net.save(filename)
