# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from config import cfg
from roi_data_layer import roidb as rdl_roidb
from custom.net import Net
#from datasets.imdb import imdb
from utils.timer import Timer

import numpy as np
import os

from custom.solver import Solver

# some models I learned and used
#bp = '/home/aphex/Projects/DeepLearning/seminar_6/output/rcnn/voc_2007_trainval/snapshot_iter_5000.pkl'
#bp = '/home/aphex/Projects/DeepLearning/seminar_6/caffe_reference.pkl'

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, roidb, output_dir):
        print "==========Initializing the SolverWrapper=========="
        self.output_dir = output_dir

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'
            
        self.roidb = roidb 
        self.solver = Solver(self.roidb, Net(bp))
        
        ################ You MIGHT want to instantiate your custom solver here.
        # Don't forget to supply roidb to the ROIPoolingLayer!
        # You should have the following line:
        # self.solver = Solver()

    def snapshot(self):
        """Saves the state the solver state."""

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.pkl')
        filename = os.path.join(self.output_dir, filename)

        self.solver.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()

            self.solver.step()
            
            timer.toc()
            if self.solver.iter % (10 * self.solver.display_freq) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(roidb, output_dir, max_iters=5000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(roidb, output_dir)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
