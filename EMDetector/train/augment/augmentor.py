"""Provides data augmentation"""
import numpy as np

from .flip import flip_augment
from .rotate90 import rotate90_augment
from .contrast import contrast_augment


class Augmentor:
  def __init__(self, params):
    self.params = params
    self._init_params()

  def _init_params(self):
    augs = ['blur', 'box', 'circle', 'elastic_warp', 'flip', 'grey',
            'misalign_slip', 'misalign_translation', 'missing_section',
            'noise', 'rotate', 'rotate90', 'rescale', 'sin']
    for aug in augs:
      if aug not in self.params.keys():
        self.params[aug] = False

  def __call__(self, sample):
    return self.augment(sample)

  def augment(self, sample):
    """Augments example.

    Args:
      img: (np array: <z,y,x,ch>) image
      labels: list of (int np array: <z,y,x,ch>), pixelwise labeling of image
      params: dict containing augmentation parameters, see code for details

    Returns:
      augmented img: image after augmentation
      augmented labels: labels after augmentation

    Note:
      augmented img,labels may not be same size as input img,labels
        because of warping
    """
    params = self.params

    # Flip
    if params['flip']:
      sample = flip_augment(sample)

    # Rotate
    if params['rotate90']:
      sample = rotate90_augment(sample)

    # Contrast change
    if params['contrast']:
      sample = contrast_augment(sample)

    # Return
    return sample