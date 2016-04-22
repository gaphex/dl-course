################################################# You MIGHT need these imports.
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
    print('==========Initializing NNet==========')
    self.net = reference_model()
    self.inp = T.tensor4('input')
    self.load(snapshot_path)    
    self.net['fc8'] = lasagne.layers.DenseLayer(self.net['fc7'],num_units=21,nonlinearity=lasagne.nonlinearities.softmax)
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
    return lasagne.layers.get_output(self.out, self.inp, deterministic=True)

  @property
  def params(self):
    return lasagne.layers.get_all_params(self.out)

  @property
  def param_values(self):
    return lasagne.layers.get_all_param_values(self.out)
