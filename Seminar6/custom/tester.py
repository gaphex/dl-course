from net import Net
from custom import utilities
import theano

class Tester(object):
  def __init__(self, snapshot_path):
    # The original Girshick's code requires this field to exist.
    self.name = ''
    self.net = Net(snapshot_path)
    self.predfn = theano.function([self.net.inp], self.net.prediction, allow_input_downcast=True)

  def forward(self, data, rois): 
    
    X = utilities.roi_layer(data, rois)
    print X.shape
    net_output = self.predfn(X)
    output = {'cls_prob': net_output}

    return output
