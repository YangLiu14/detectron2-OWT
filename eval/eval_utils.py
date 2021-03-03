import cv2
import glob
import json
import numpy as np
import os
import pickle
import png
import sys
import time
import torch
import tqdm


from detectron2.data.detection_utils import read_image
from pycocotools.mask import encode, decode, area, toBbox
from detectron2.structures.instances import Instances
from typing import List
from itertools import groupby


def bbox_iou(boxA, boxB):
    """
    bbox in the form of [x1,y1,x2,y2]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# ==============================================================
# G-IoU, adapted from:
# https://github.com/generalized-iou/Detectron.pytorch/blob/master/lib/utils/net.py
# =============================================================
def compute_giou(box1, box2, bbox_inside_weights, bbox_outside_weights):

    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(box1)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)
    # iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    # iouk = ((1 - iouk) * iou_weights).sum(0)
    # miouk = ((1 - miouk) * iou_weights).sum(0)

    # return iouk, miouk
    return miouk


def open_flow_png_file(file_path_list):
    # Decode the information stored in the filename
    flow_png_info = {}
    for file_path in file_path_list:
        file_token_list = os.path.splitext(file_path)[0].split("_")
        minimal_value = int(file_token_list[-1].replace("minimal", ""))
        flow_axis = file_token_list[-2]
        flow_png_info[flow_axis] = {'path': file_path,
                                    'minimal_value': minimal_value}

    # Open both files and add back the minimal value
    for axis, flow_info in flow_png_info.items():
        png_reader = png.Reader(filename=flow_info['path'])
        flow_2d = np.vstack(list(map(np.float32, png_reader.asDirect()[2])))

        # Add the minimal value back
        flow_2d = flow_2d.astype(np.float32) + flow_info['minimal_value']

        flow_png_info[axis]['flow'] = flow_2d

    # Combine the flows
    flow_x = flow_png_info['x']['flow']
    flow_y = flow_png_info['y']['flow']
    flow = np.stack([flow_x, flow_y], 2)

    return flow


def warp_flow(img, flow, binarize=True):
    """
    Use the given optical-flow vector to warp the input image/mask in frame t-1,
    to estimate its shape in frame t.
    :param img: (H, W, C) numpy array, if C=1, then it's omissible. The image/mask in previous frame.
    :param flow: (H, W, 2) numpy array. The optical-flow vector.
    :param binarize:
    :return: (H, W, C) numpy array. The warped image/mask.
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if binarize:
        res = np.equal(res, 1).astype(np.uint8)
    return res


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


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
    """
    Store all the proposals in one frame.
    Output json format:

    - List of Dict:
        {"category_id": (int),
         "bbox": [x1, y1, x2, y2],
         "instance_mask": {"size": [img_h, img_w], "counts": rle_str}
         "score": (float),
         "bg_score": (float),
         "objectness": (float),
         "embeddings": (np.array), shape=(2048,)
    """

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
            proposal['bbox'] = [float(b) for b in bbox]  # Convert bbox coordinates to int
            # Convert mask(numpy array) to mask(RLE)
            mask = predictions['instances'].pred_masks[i].cpu().numpy()
            mask_rle = encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            mask_rle['counts'] = mask_rle['counts'].decode(encoding="utf-8")
            proposal['instance_mask'] = mask_rle
            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolt()
            proposal['bg_score'] = predictions['instances'].bg_scores[i].cpu().numpy().tolist()
            proposal['objectness'] = predictions['instances'].objectness[i].cpu().numpy().tolist()
            # proposal['embeddings'] = predictions['instances'].embeddings[i].cpu().numpy().tolist()
            embeddings = predictions['instances'].embeddings[i].cpu().numpy().astype(np.float32)
            proposal['embeddings'] = embeddings.tolist()
            # proposal['embeddings'] = [float(e) for e in proposal['embeddings']]
            # output['proposals'].append(proposal)
            output.append(proposal)
    # output['sem_seg'] = predictions['sem_seg'].cpu().numpy()

    with open(json_outpath, 'w') as fout:
        json.dump(output, fout)


def store_TAOnpz(predictions, input_img_path: str, valid_classes: List[int], npz_outdir: str):
    """
    Same as store_TAOjson, but store in .npz file --> 5 times smaller.
    """
    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.npz')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    npz_outpath = os.path.join(npz_outdir, frame_name)
    # output = dict()
    # output['proposals'] = list()
    output = list()

    pred_classes = predictions['instances'].pred_classes

    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            bbox = predictions['instances'].pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            proposal['bbox'] = [float(b) for b in bbox]  # Convert bbox coordinates to int
            # Convert mask(numpy array) to mask(RLE)
            mask = predictions['instances'].pred_masks[i].cpu().numpy()
            mask_rle = encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            mask_rle['counts'] = mask_rle['counts'].decode(encoding="utf-8")
            proposal['instance_mask'] = mask_rle
            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolist()
            proposal['bg_score'] = predictions['instances'].bg_scores[i].cpu().numpy().tolist()
            proposal['objectness'] = predictions['instances'].objectness[i].cpu().numpy().tolist()
            embeddings = predictions['instances'].embeddings[i].cpu().numpy()
            # Each value in the shaped (2048,) embedding is between [0, 1]
            # Each value * 1e4 and store as uint16 will save a lot of storage space,
            # when using the embeddings, each value should be again divided by 1e4.
            # This encode/decode operation is equal to keeping 4 decimal digits of the original float value.
            #   uint16: Unsigned integer (0 to 65535)
            proposal['embeddings'] = (embeddings * 1e4).astype(np.uint16).tolist()

            # output['proposals'].append(proposal)
            output.append(proposal)
    # output['sem_seg'] = predictions['sem_seg'].cpu().numpy()

    np.savez_compressed(npz_outpath, output)


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


def remove_mask_overlap(proposals):
    """
    Args:
        proposals: List[Dict], sorted proposals according specific scoring criterion. Each proposal contains:
        {
            category_id: int,
            bbox: [x1, y1, x2, y2],
            score: float (could be named differently, e.g. bg_score, objectness, etc)
            instance_mask: COCO_RLE format,
        }

    Returns:
        selected_props, List[Dict]
    """
    masks = [decode(prop['instance_mask']) for prop in proposals]
    idx = [i for i in range(len(proposals))]
    labels = np.arange(1, len(proposals) + 1)
    png = np.zeros_like(masks[0])

    # Put the mask there in reversed order, so that the latter one would just cover the previous one,
    # and the latter one has higher score. (Because proposals are sorted)
    for i in reversed(range(len(proposals))):
        png[masks[i].astype("bool")] = labels[i]

    refined_masks = [(png == id_).astype(np.uint8) for id_ in labels]
    refined_segmentations = [encode(np.asfortranarray(refined_mask)) for refined_mask in refined_masks]
    selected_props = []
    for prop, refined_segmentation, mask in zip(proposals, refined_segmentations, refined_masks):
        refined_segmentation['counts'] = refined_segmentation['counts'].decode("utf-8")
        if area(refined_segmentation) == 0:
            continue
        prop['instance_mask'] = refined_segmentation
        box = toBbox(refined_segmentation).tolist()  # in the form of [xc, yc, w, h]
        # convert [xc, yc, w, h] to [x1, y1, x2, y2]
        bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        prop['bbox'] = bbox
        # prop['mask'] = mask

        selected_props.append(prop)

    return selected_props


def remove_mask_overlap_small_on_top(proposals):
    """
    Similar with `remove_mask_overlap`, but instead of putting masks with higher scores on top,
    put masks with smaller area on top.

    Args:
        proposals: List[Dict], sorted proposals according specific scoring criterion. Each proposal contains:
        {
            category_id: int,
            bbox: [x1, y1, x2, y2],
            score: float (could be named differently, e.g. bg_score, objectness, etc)
            instance_mask: COCO_RLE format,
        }

    Returns:
        selected_props, List[Dict]
    """
    # Sort proposals by it's area.
    # all_areas = [area(prop['instance_mask']) for prop in proposals]
    proposals.sort(key=lambda prop: area(prop['instance_mask']))
    # all_areas2 = [area(prop['instance_mask']) for prop in proposals]

    masks = [decode(prop['instance_mask']) for prop in proposals]
    idx = [i for i in range(len(proposals))]
    labels = np.arange(1, len(proposals) + 1)
    png = np.zeros_like(masks[0])

    # Put the mask there in reversed order, so that the latter one would just cover the previous one,
    # and the latter one has smaller scores. (Because proposals are sorted by area)
    for i in reversed(range(len(proposals))):
        png[masks[i].astype("bool")] = labels[i]

    refined_masks = [(png == id_).astype(np.uint8) for id_ in labels]
    refined_segmentations = [encode(np.asfortranarray(refined_mask)) for refined_mask in refined_masks]
    selected_props = []
    for prop, refined_segmentation, mask in zip(proposals, refined_segmentations, refined_masks):
        refined_segmentation['counts'] = refined_segmentation['counts'].decode("utf-8")
        if area(refined_segmentation) == 0:
            continue
        prop['instance_mask'] = refined_segmentation
        box = toBbox(refined_segmentation).tolist()  # in the form of [xc, yc, w, h]
        # convert [xc, yc, w, h] to [x1, y1, x2, y2]
        bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        prop['bbox'] = bbox
        # prop['mask'] = mask

        selected_props.append(prop)

    return selected_props


if __name__ == "__main__":
    # root_dir = "/home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/postNMS_bbox/Panoptic_Cas_R101_NMSoff+objectness003_objectness"
    root_dir = "/home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS/_objectness/"
    outdir = "/home/kloping/OpenSet_MOT/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/nonOverlap_small/objectness/"
    scoring = "objectness"
    # Read in proposals
    # data_srcs = ["ArgoVerse", "BDD", "Charades",  "LaSOT", "YFCC100M"]
    data_srcs = ["ArgoVerse"]
    for data_src in data_srcs:
        print("Processing", data_src)
        videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, data_src, '*')))]
        for idx, video in enumerate(tqdm.tqdm(videos)):
            fpath = os.path.join(root_dir, data_src, video)
            json_files = sorted(glob.glob(fpath + '/*' + '.json'))
            for jpath in json_files:
                json_name = jpath.split("/")[-1]
                with open(jpath, 'r') as f:
                    proposals = json.load(f)
                selected_proposals = remove_mask_overlap_small_on_top(proposals)
                selected_proposals.sort(key=lambda prop: prop[scoring])

                # Store new json file
                outpath = os.path.join(outdir, data_src, video, json_name)
                if not os.path.exists(os.path.join(outdir, data_src, video)):
                    os.makedirs(os.path.join(outdir, data_src, video))
                with open(outpath, 'w') as f:
                    json.dump(selected_proposals, f)
