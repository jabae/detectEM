import numpy as np


def translate_augment(sample):
  """Performs translation augmentation on img.

  Args:
    sample: Dictionary of (np array: <ch,z,x,y>) image and mask
  """
  
  w_all = sample["image"].shape[0]
  w = w_all//2

  tw = np.random.randint(-w//2, w//2)
  th = np.random.randint(-w//2, w//2)

  for k in sample.keys():  
    sample[k] = sample[k][:,w_all//2-tw:w_all//2-tw+w,
                          w_all//2-th:w_all//2-th+w]
    
    
  return sample