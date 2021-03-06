# pretrained weights are available at https://www.dropbox.com/s/mlbzu8nnfo80rbd/snapshot_iter_250.pkl?dl=1

import cPickle as pickle
import lasagne
from models import reference_model, vgg16_model
import theano.tensor as T

class Net(object):
  """A class for holding a symbolic representation of the neural network.
  Instances of this class are going to be used both in the solver and 
  in the tester. 
  """

  def __init__(self, snapshot_path=None):
    print('Initializing Net')
    self.net = reference_model()
    self.inp = T.tensor4('input', dtype='float32')
    
    if 'caffe_reference' in snapshot_path: self.load(snapshot_path) #we load reference weights **before** we modify the net
    self.patch_net()
    if 'caffe_reference' not in snapshot_path: self.load(snapshot_path)   # But normally we do it after
    
  def patch_net(self):
    print 'Patching Net'
    self.net['fc7_dropout'] = lasagne.layers.DropoutLayer(self.net['fc7'], p=0.2)
    self.net['fc8'] = lasagne.layers.DenseLayer(self.net['fc7_dropout'],num_units=21,nonlinearity=lasagne.nonlinearities.softmax)
    self.out = self.net['fc8']
    
  def save(self, filename):
    weights = lasagne.layers.get_all_param_values(self.out)
    pickle.dump(weights, open(filename, 'w'),protocol=pickle.HIGHEST_PROTOCOL)
    
  def load(self, snapshot_path):
    if snapshot_path:
      lasagne.layers.set_all_param_values(self.net.values(), pickle.load(open(snapshot_path, 'r')))
      print 'loaded weights from', snapshot_path

  @property
  def input(self):
    return self.inp

  @property
  def prediction(self):
    return lasagne.layers.get_output(self.out, self.inp, deterministic=False)

  @property
  def params(self):
    return lasagne.layers.get_all_params(self.out)

  @property
  def param_values(self):
    return lasagne.layers.get_all_param_values(self.out)
