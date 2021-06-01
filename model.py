import sys
def add_path_to_sys(dp_path):
  if dp_path not in sys.path:
    sys.path.append(dp_path)

add_path_to_sys('./detectron2_repo/projects/DensePose')

from detectron2.config import get_cfg
from densepose.vis.extractor import DensePoseResultExtractor
from densepose import add_densepose_config
from detectron2.engine.defaults import DefaultPredictor

import numpy as np
import io

class DensePoseEstimator:
  def __init__(self):
        config_fpath = "./detectron2_repo/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        model_fpath  = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        
        cfg.merge_from_list([])
        cfg.merge_from_list(['MODEL.ROI_HEADS.SCORE_THRESH_TEST', '0.8'])
        
        
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze() 
        self.predictor = DefaultPredictor(cfg)
        
  def __call__(self, data):
    image = np.load(io.BytesIO(data))
    outputs = self.predictor(image)["instances"]

    extractor = DensePoseResultExtractor()
    data = extractor(outputs)
    #import pdb; pdb.set_trace()
    mask = data[0][0].labels.cpu().numpy()
    bbox = data[1].cpu().numpy()
    
    x, y, w, h = bbox[0]
    x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
    #h_min = np.minimum(h-y, mask.shape[0])
    #w_min = np.minimum(w-x, mask.shape[1])

    big_mask = np.zeros((image.shape[0], image.shape[1]))
    big_mask[y:y+h, x:x+w] = mask
    
    return big_mask        
