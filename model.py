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
      uv = (
              quantize_densepose_chart_result(data[0][0])
              .to("cpu")
              .labels_uv_uint8.permute(1, 2, 0)
              .numpy()
          )

      bbox = data[1][0].cpu().numpy()
      x, y = int(bbox[0]), int(bbox[1])
      h, w = data[0][0].labels.shape

      uv_original = np.zeros_like(image)
      uv_original[y: y + h, x: x + w] = uv
      return uv_original
