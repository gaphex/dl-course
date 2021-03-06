################################################# You MIGHT need these imports.
import sys
from net import Net
from copy import deepcopy

from fast_rcnn.config import cfg
from roi_data_layer.layer import RoIDataLayer
from lasagne.regularization import regularize_layer_params, l1, l2
from custom.utilities import *
import theano
import lasagne
import theano.tensor as T

class Solver(object):
  def __init__(self, roidb, net, freeze=0):
    # Holds current iteration number. 
    self.iter = 0

    # How frequently we should print the training info.
    self.display_freq = 1

    # Holds the path prefix for snapshots.
    self.snapshot_prefix = 'snapshot'
    
    self.roidb = roidb
    self.net = net
    self.freeze = freeze
    self.roi_data_layer = RoIDataLayer()
    self.roi_data_layer.setup()
    self.roi_data_layer.set_roidb(self.roidb)
    self.stepfn = self.build_step_fn(self.net)
    self.predfn = self.build_pred_fn(self.net)

  # This might be a useful static method to have.
  #@staticmethod not so static anymore
  def build_step_fn(self, net):
    target_y = T.vector("target Y",dtype='int64')
    tl = lasagne.objectives.categorical_crossentropy(net.prediction,target_y)
    loss = tl.mean()
    accuracy = lasagne.objectives.categorical_accuracy(net.prediction,target_y).mean()
   
    weights = net.params
    grads = theano.grad(loss, weights)
    
    scales = np.ones(len(weights))

    if self.freeze:
        scales[:-self.freeze] = 0
        
    print 'GRAD SCALE >>>', scales
    
    for idx, param in enumerate(weights):
        grads[idx] *= scales[idx]
        grads[idx] = grads[idx].astype('float32')
        
    #updates_sgd = lasagne.updates.sgd(loss, net.params, learning_rate=0.0001)
    updates_sgd = lasagne.updates.sgd(grads, net.params, learning_rate=0.0001)
    
    stepfn = theano.function([net.inp, target_y], [loss, accuracy], updates=updates_sgd, allow_input_downcast=True)
    return stepfn

  @staticmethod
  def build_pred_fn(net):
    predfn = theano.function([net.inp], net.prediction, allow_input_downcast=True)
    return predfn

  def get_training_batch(self):
    """Uses ROIDataLayer to fetch a training batch.

    Returns:
      input_data (ndarray): input data suitable for R-CNN processing
      labels (ndarray): batch labels (of type int32)
    """
    data, rois, labels = deepcopy(self.roi_data_layer.top[: 3])
    X = roi_layer(data, rois)
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
