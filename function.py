import io
import numpy as np

from model import DensePoseEstimator

model = DensePoseEstimator()

def call(img):
	segmentation_mask = model(img)
	bytes_io = io.BytesIO()
	np.save(bytes_io, segmentation_mask, allow_pickle=False)
	return bytes_io.getvalue()
