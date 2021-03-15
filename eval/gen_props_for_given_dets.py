import argparse
import glob
import multiprocessing as mp
import os
import time
import json
import cv2
import tqdm
import torch
import numpy as np
from pycocotools.mask import encode, decode, area, toBbox
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo


class SegmentedObject:
    def __init__(self, bbox, mask, class_id, track_id, frame_id, video_name):
        self.bbox = bbox
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id
        self.frame_id = frame_id
        self.video_name = video_name


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # from detectron2 import model_zoo
    # cfg.merge_from_file(model_zoo.get_config_file(
    #     "COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_200_FPN_syncBN_all_tricks_3x.yaml"))
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    # TEST
    # cfg.MODEL.WEIGHTS = "/nfs/cold_project/liuyang/mots1/ResNeSt_finetune/noBackBone_COCO_dataAug/model_0007999.pth"
    # cfg.MODEL.WEIGHTS = "/nfs/cold_project/liuyang/detectron2_weights/" \
    #                     "mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x-e1901134.pth"
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # END of TEST

    cfg.freeze()
    return cfg


def load_txt(path):
    # 0          1          2          3       4       5
    # <frame_id> <track_id> <class_id> <img_h> <img_w> <rle>
    props_per_frame = dict()
    video_name = path.split('/')[-1].replace(".txt", "")
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()

            frame_id, track_id, class_id = int(fields[0]), int(fields[1]), int(fields[2])
            if frame_id not in props_per_frame.keys():
                props_per_frame[frame_id] = []

            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            if area(mask) == 0:
                continue
            bbox = toBbox(mask).tolist()

            props_per_frame[frame_id].append(
                {'bbox': bbox,  # [x, y, w, h]
                 'mask': mask,
                 'class_id': class_id,
                 'track_id': track_id,
                 'frame_id': frame_id,
                 'video_name': video_name,
                 'embeddings': None
                }
            )

    return props_per_frame, video_name


def main(model, image_dir, det_dir, outdir):
    txt_fpaths = sorted(glob.glob(det_dir + '/*' + '.txt'))
    for fpath in txt_fpaths:
        props_per_frame, video_name = load_txt(fpath)
        video_dir = os.path.join(image_dir, video_name)
        image_paths = sorted(glob.glob(video_dir + '/*' + '.jpg'))

        for img_idx, img_path in enumerate(image_paths):
            props = props_per_frame[img_idx]
            img = read_image(img_path, format="BGR")
            bboxes = np.array([p['bbox'] for p in props])  # convert xywh to x1y1x2y2
            predictions_all = model.predictor(img, bboxes)
            for prop in props:
                x, y, w, h = prop['bbox']
                curr_box = [x, y, x+w, y+h]
                predictions = model.predictor(img, np.array([prop['bbox']]))
                import pdb; pdb.set_trace()
                print('tbc')

        curr_outdir = os.path.join(outdir, video_name)
        if not os.path.exists(curr_outdir):
            os.makedirs(curr_outdir)


        import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Proposal Similarity')
    parser.add_argument('--image_dir', required=True, type=str, help='Root directory stores all the images')
    parser.add_argument('--det_dir', required=True, type=str, help='Root directory stores all the detections in txt')
    parser.add_argument('--outdir', type=str, help='Output directory of intermediate results')

    # Detectron2 related args
    parser.add_argument("--config-file", default="../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml",
        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Set up model
    ALL_START = time.time()
    print("Start counting time. ALL_START")
    # mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    image_dir = args.image_dir

    if args.outdir:
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir = None

    main(demo, args.image_dir, args.det_dir, args.outdir)

