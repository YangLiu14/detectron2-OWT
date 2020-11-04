
import cv2
import json
import numpy as np
import os
import sys
import time
import torch
import tqdm


from detectron2.data.detection_utils import read_image
from pycocotools.mask import encode, decode
from detectron2.structures.instances import Instances
from typing import List
from itertools import groupby


def filter_pred_by(valid_classes: List[int], predictions):
    pred_classes = predictions.pred_classes
    # print("num of pred_classes before:", len(pred_classes))
    bboxes = []
    scores = []
    new_pred_cls = []
    pred_masks = []

    for i in range(len(pred_classes)):
        if pred_classes[i] in valid_classes:
            bboxes.append(predictions.pred_boxes[i].tensor[0])

            scores.append(predictions.scores[i])
            new_pred_cls.append(predictions.pred_classes[i])
            pred_masks.append(predictions.pred_masks[i])

    if len(bboxes) == 0:
        return Instances(image_size=(0, 0)), True

    predictions.pred_boxes.tensor = torch.stack(bboxes, dim=0)
    predictions.scores = torch.Tensor(scores)
    predictions.pred_classes = torch.Tensor(new_pred_cls)
    predictions.pred_masks = torch.stack(pred_masks, dim=0)
    # print("num of pred_classes before:", len(new_pred_cls))
    return predictions, False


def mask2polygon(mask):
    """Convert binary mask to polygon (COCO format)
    :param mask: numpy array
    :return: list (in the format of polygon)
    """
    segmentation = []
    contours, hierarchy = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    if len(segmentation) == 0:
        print('error in cv2.findContours')

    return segmentation


def store_coco(predictions, input_img_path: str, valid_classes: List[int], json_outdir: str):
    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.json')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    json_outpath = os.path.join(json_outdir, frame_name)
    output = []

    pred_classes = predictions['instances'].pred_classes
    res = dict()
    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            proposal['bbox'] = predictions['instances'].pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolist()
            # Convert Mask to Polygon format
            mask = predictions['instances'].pred_masks[i].cpu().numpy()
            res['image_shape'] = mask.shape
            mask_polygon = mask2polygon(mask)
            proposal['segmentation'] = mask_polygon
            proposal['area'] = int(mask.sum())
            output.append(proposal)

    res['props'] = output
    return res


# In the style of UnOVOST
def store_json(predictions, input_img_path: str, valid_classes: List[int], json_outdir: str):

    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.json')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    json_outpath = os.path.join(json_outdir, frame_name)
    output = []

    pred_classes = predictions['instances'].pred_classes

    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            proposal['bbox'] = predictions['instances'].pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolist()
            # Convert Mask to RLE format
            mask = predictions['instances'].pred_masks[i].cpu().numpy()
            mask_rle = encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            mask_rle['counts'] = mask_rle['counts'].decode(encoding="utf-8")
            proposal['segmentation'] = mask_rle
            # forward seg, left empty for now
            fwd_seg = dict()
            fwd_seg['size'] = proposal['segmentation']['size']
            fwd_seg['counts'] = None
            proposal['forward_segmentation'] = fwd_seg
            proposal['backward_segmentation'] = None
            proposal['ReID'] = None
            output.append(proposal)

    with open(json_outpath, 'w') as fout:
        json.dump(output, fout)


# In the style of TAO
def store_TAOjson(predictions, input_img_path: str, valid_classes: List[int], json_outdir: str):

    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.json')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    json_outpath = os.path.join(json_outdir, frame_name)
    # output = dict()
    # output['proposals'] = list()
    output = list()

    pred_classes = predictions['instances'].pred_classes

    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            bbox = predictions['instances'].pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            proposal['bbox'] = [int(b) for b in bbox]  # Convert bbox coordinates to int
            # Convert mask(numpy array) to mask(RLE)
            mask = predictions['instances'].pred_masks[i].cpu().numpy()
            mask_rle = encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            mask_rle['counts'] = mask_rle['counts'].decode(encoding="utf-8")
            proposal['instance_mask'] = mask_rle
            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolist()
            proposal['bg_score'] = predictions['instances'].bg_scores[i].cpu().numpy().tolist()
            proposal['objectness'] = predictions['instances'].objectness[i].cpu().numpy().tolist()
            # proposal['embeddings'] = predictions['instances'].embeddings[i].cpu().numpy().tolist()
            # output['proposals'].append(proposal)
            output.append(proposal)
    # output['sem_seg'] = predictions['sem_seg'].cpu().numpy()

    with open(json_outpath, 'w') as fout:
        json.dump(output, fout)



def analyse_coco_cat(predictions, input_img_path: str, valid_classes: List[int], json_outdir: str):
    """
    Anaylse the output of the network to see if there is any coco classes id greater than 80.
    """

    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.json')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    json_outpath = os.path.join(json_outdir, frame_name)
    output = dict()
    output['proposals'] = list()

    pred_classes = predictions['instances'].pred_classes

    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            if proposal['category_id'] > 80:
                print(proposal['category_id'])
                sys.exit()

    # output['sem_seg'] = predictions['sem_seg'].cpu().numpy()

    with open(json_outpath, 'w') as fout:
        json.dump(output, fout)


def image_stitching(image_paths, rows, cols, out_path):
    """
    Stitch list of images into (rows x cols) image-tiles.
    Args:
        image_paths: List of image paths, ordered in the fashion that, the 1st image with be placed at (0,0)
                     in the resulting image tile, the 2nd image at(0,1), 3rd at (0,2) and so on.
        rows: Int, number of rows in the resulting image tile.
        cols: Int, number of columns in the resulting image tile.
        out_path: Str, output path and file name.
    """
    assert rows * cols == len(image_paths)
    # Read images as numpy arrays
    img_list = [cv2.imread(filename) for filename in image_paths]
    img_shapes = np.array([np.array(im.shape) for im in img_list])
    H, W = min(img_shapes[:, 0]), min(img_shapes[:, 1])
    img_list = [im[:H, :W, :] for im in img_list]


    # combine images vertically
    img_vert = list()

    while img_list:
        curr_row = img_list[:rows]
        img_list = img_list[rows:]
        combined_img = np.hstack(curr_row)
        img_vert.append(combined_img)
    # combine images horizontally
    all_combined = np.vstack(img_vert)

    # Save image
    cv2.imwrite(out_path, all_combined)
