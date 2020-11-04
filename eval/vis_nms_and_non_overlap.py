import argparse
import cv2
import glob
import json
import numpy as np
import os
import random
import torch
import tqdm

from collections import defaultdict
from pycocotools.mask import encode, decode, toBbox
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from eval.eval_recall_vs_nprops import remove_mask_overlap

"""
Folder structure:

- json_dir/
    - base + "_score"/
        - ArgoVerse/
            - video1/
                - frame1.json
                - frame2.json
                - ...
            - video2/
            - .../
        - BDD/
        - Charades/
        - LaSOT/
        - YFCC100M/
    - base + "_bg_score"/
    - .../

- img_dir/
    - ArgoVerse/
        - video1/
            - frame1.jpg
            - frame2.jpg
            - ...
        - video2/
        - .../
    - BDD/
    - Charades/
    - LaSOT/
    - YFCC100M/
"""


def vis_one_frame(img_path: str, img_idx: int, frame_path: str, outdir: str, scoring: str, conf_threshold: float):
    if not os.path.exists(outdir + '/' + scoring):
        os.makedirs(outdir + '/' + scoring)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    img_shape = (img.shape[0], img.shape[1])  # HxW

    # Load json
    with open(frame_path, 'r') as f:
        proposals = json.load(f)

    # postNMS visualizatoin
    all_boxes = [prop['bbox'] for prop in proposals]
    all_boxes = torch.from_numpy(np.vstack(all_boxes))
    all_masks = [prop['instance_mask'] for prop in proposals]
    all_scores = [prop[scoring] for prop in proposals]
    all_classes = np.ones((len(proposals), ))

    # Convert to detectron2 recognizable prediction
    pred_nms = Instances(img_shape)
    pred_nms.pred_boxes = Boxes(all_boxes)   # boxes : Nx4 numpy array (x1, y1, x2, y2)
    pred_nms.scores = all_scores
    pred_nms.pred_masks = all_masks
    pred_nms.pred_classes = all_classes

    # Non-Overlapping
    non_overlap_props = remove_mask_overlap(proposals)

    all_boxes = [prop['bbox'] for prop in non_overlap_props]
    all_boxes = torch.from_numpy(np.vstack(all_boxes))
    all_masks = [prop['instance_mask'] for prop in non_overlap_props]
    all_scores = [prop[scoring] for prop in non_overlap_props]
    all_classes = np.ones((len(non_overlap_props),))

    pred_non_overlap = Instances(img_shape)
    pred_non_overlap.pred_boxes = Boxes(all_boxes)   # boxes : Nx4 numpy array (x1, y1, x2, y2)
    pred_non_overlap.scores = all_scores
    pred_non_overlap.pred_masks = all_masks
    pred_non_overlap.pred_classes = all_classes

    # visualization
    name_components = [img_path.split("/")[-3]] + \
                      [str(img_idx), "_nms", str(len(pred_nms)), "vs.", "nonOverlap", str(len(pred_non_overlap))]
    img_name = "".join(name_components) + '.jpg'
    vis = Visualizer(img)
    vis_pred_nms = vis.draw_instance_predictions(pred_nms).get_image()
    vis_pred_non_overlap = vis.draw_instance_predictions(pred_non_overlap).get_image()

    concat = np.concatenate((vis_pred_nms, vis_pred_non_overlap), axis=1)
    cv2.imwrite(os.path.join(outdir, scoring, img_name), concat[:, :, ::-1])


def main(img_dir: str, json_dir: str, outdir: str, conf_threshold: float):

    base = "Panoptic_Cas_R101_NMSoff+objectness003"
    scorings = ["bg_rpn_product", "bg_rpn_sum", "bg_score", "objectness", "one_minus_bg_score", "score"]

    for scoring in scorings:
        print(">>>>>>>>>  Evaluating", scoring)
        root_dir = os.path.join(json_dir, base + "_" + scoring)

        video_src_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
        print(">>>>>>>>> Processing the following datasets: {}".format(video_src_names))

        for video_src in video_src_names:
            print("Processing", video_src)
            video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, video_src, '*')))]
            video_names.sort()

            # Randomly pick 5 of the videos, and in each picked video, randomly choose one frame to
            # do non-overlapping and visualize the result.
            idx = random.sample(range(len(video_names)), 5)
            for i in idx:
                # Extract frame_path
                video_name = video_names[i]
                all_frames = glob.glob(os.path.join(root_dir, video_src, video_names[i], "*.json"))
                frame_idx = random.randint(0, len(all_frames) - 1)
                frame_path = all_frames[frame_idx]
                # Extract img_path
                img_path = os.path.join(img_dir, '/'.join(frame_path.split("/")[-3:]).replace(".json", ".jpg"))
                # Perform visualization
                vis_one_frame(img_path, i, frame_path, outdir, scoring, conf_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions after NMS operation and non-overlap operation."
    )
    parser.add_argument("--img_dir", required=True, help="Directory of images")
    parser.add_argument("--json_dir", required=True, help="Directory of post-NMS json output")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--conf_threshold", default=0.0, type=float, help="confidence threshold")
    args = parser.parse_args()

    # TODO: also add gt there
    main(args.img_dir, args.json_dir, args.outdir, args.conf_threshold)


