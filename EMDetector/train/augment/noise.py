import numpy as np


def noise_augment(sample):
	"""Performs noise addition augmentation on img.

	Args:
		sample: Dictionary of (np array: <ch,z,x,y>) image and mask
	"""

	b = np.random.randint(-4,4)/255

	sample["image"] = sample["image"] + b
	sample["image"] = np.clip(sample["image"], 0, 1)


	return sample