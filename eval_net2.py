#!/usr/bin/env python

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
register_coco_instances("tao_val", {}, "datasets/tao/annotations/validation.json", "/Volumes/Elements1T/TAO_VAL/")



def setup():
    """
    Create configs and perform basic setups.
    """
    config_file = "configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    cfg.MODEL.WEIGHTS = "/Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl"  # path to the model we just trained
    cfg.MODEL.DEVICES = "cpu"

    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # default_setup(cfg, args)
    return cfg


def main():
    cfg = setup()




main()