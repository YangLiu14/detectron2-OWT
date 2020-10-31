import argparse
import glob
import json
import math
import numpy as np
import os
import tqdm

from pycocotools.mask import encode, decode, iou


# https://www.programmersought.com/article/97214443593/
def nms_bbox(bounding_boxes, confidence_score, threshold=0.5):
    """
    Args:
        bounding_boxes: List, Object candidate bounding boxes
        confidence_score: List, Confidence score of the bounding boxes
        threshold: float, IoU threshold

    Returns:
        List, bboxes and scores that remains
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        a = start_x[index]
        b = order[:-1]
        c = start_x[order[:-1]]
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]


    # TEST: ensure that the picked scores are in descending order
    for i in range(1, len(picked_score)):
        if picked_score[i-1] < picked_score[i]:
            msg = "{} at index {} should not be smaller than {} at index {}.".format(picked_score[i-1], i-1,
                                                                                     picked_score[i], i)
            raise Exception(msg)
    # END of TEST

    return picked_boxes, picked_score


def nms_mask(masks, confidence_score, threshold=0.5):
    """
    Args:
        masks: List, each instance mask is in the form of RLE.
        confidence_score: List, Confidence score of the masks
        threshold: float, IoU threshold

    Returns:
        List, masks and scores that remains
    """
    # If no bounding boxes, return empty list
    if len(masks) == 0:
        return [], []

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_masks = []
    picked_score = []

    # Sort by confidence score of masks
    order = np.argsort(score)

    remained_masks = masks.copy()  # masks remains to be evaluated
    remained_scores = confidence_score.copy() # masks remains to be evaluated
    last_len = -1
    while True:
        # The index of largest confidence score
        index = order[-1]

        # Pick the mask with largest confidence score
        picked_masks.append(remained_masks[index])
        picked_score.append(remained_scores[index])

        # Compare the IoUs of the rest of the masks with current mask
        iscrowd_flags = [int(True)] * len(remained_masks)
        ious = iou([picked_masks[-1]], remained_masks, pyiscrowd=iscrowd_flags)  # TODO: is pyiscrowd correctly set?
        ious = ious.squeeze()
        remained_idx = np.where(ious < threshold)[0]
        if remained_idx.size == last_len:   # There are no masks overlap too much with the current one
            remained_masks.pop(index)
            remained_scores.pop(index)
            score = np.array(remained_scores)
            order = np.argsort(score)
            last_len = remained_idx.size
            continue
        if remained_idx.size == 0:
            picked_masks += remained_masks
            picked_score += remained_scores
            break
        last_len = remained_idx.size

        # Update masks and their corresponding scores remained to be evaluated
        tmp = [remained_masks[i] for i in remained_idx]
        remained_masks = tmp.copy()
        tmp = [remained_scores[i] for i in remained_idx]
        remained_scores = tmp.copy()
        # Re-calculate the order
        score = np.array(remained_scores)
        order = np.argsort(score)

    # TEST: ensure that the picked scores are in descending order
    for i in range(1, len(picked_score)):
        if picked_score[i-1] < picked_score[i]:
            msg = "{} at index {} should not be smaller than {} at index {}.".format(picked_score[i-1], i-1,
                                                                                     picked_score[i], i)
            raise Exception(msg)
    # END of TEST

    return picked_masks, picked_score


def process_one_frame(seq: str, scoring: str, iou_thres: float, outpath: str):
    # Load original proposals
    with open(seq, 'r') as f:
        proposals = json.load(f)

    props_for_nms = dict()
    props_for_nms['props'] = list()
    props_for_nms['scores'] = list()

    for prop in proposals:
        cat_id = prop['category_id']
        if scoring == "one_minus_bg_score":
            curr_score = 1 - prop['bg_score']
        elif scoring == "bg_rpn_sum":
            curr_score = (1000 * prop["objectness"] + prop["bg_score"]) / 2
        elif scoring == "bg_rpn_product":
            curr_score = math.sqrt(1000 * prop["objectness"] * prop["bg_score"])
        else:
            curr_score = prop[scoring]

        props_for_nms['props'].append(prop[args.nms_criterion])
        props_for_nms['scores'].append(curr_score)

    output = list()
    if args.nms_criterion == 'bbox':
        props_nms, scores_nms = nms_bbox(props_for_nms['props'], props_for_nms['scores'], iou_thres)
    elif args.nms_criterion == 'instance_mask':
        props_nms, scores_nms = nms_mask(props_for_nms['props'], props_for_nms['scores'], iou_thres)
    else:
        raise Exception(args.nms_criterion, "invalid. Please choose from `bbox` or `mask`")

    # Class-agnostic fashion: does not output category id
    for prop, score in zip(props_nms, scores_nms):
        output.append({args.nms_criterion: prop, scoring: score})

    # Store proposals after NMS
    outdir = "/".join(outpath.split("/")[:-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outpath, 'w') as f:
        json.dump(output, f)


# TODO: this function is not yet adapted for mask_based_nms
def process_one_frame_categorywise(seq: str, scoring: str, iou_thres: float, outpath: str):
    # Load original proposals
    with open(seq, 'r') as f:
        proposals = json.load(f)

    props_for_nms = dict()

    for prop in proposals:
        cat_id = prop['category_id']
        if scoring == "one_minus_bg_score":
            curr_score = 1 - prop['bg_score']
        elif scoring == "bg_rpn_sum":
            curr_score = (1000 * prop["objectness"] + prop["bg_score"]) / 2
        elif scoring == "bg_rpn_product":
            curr_score = math.sqrt(1000 * prop["objectness"] * prop["bg_score"])
        else:
            curr_score = prop[scoring]

        if cat_id in props_for_nms.keys():
            props_for_nms[cat_id]['bboxes'].append(prop['bbox'])
            props_for_nms[cat_id]['scores'].append(curr_score)
        else:
            props_for_nms[cat_id] = {'bboxes': [prop['bbox']],
                                     'scores': [curr_score]}

    output = list()
    for cat_id, data in props_for_nms.items():
        if len(data['bboxes']) > 1:
            bboxes_nms, scores_nms = nms_bbox(data['bboxes'], data['scores'], iou_thres)
        else:
            bboxes_nms, scores_nms = data['bboxes'], data['scores']
        for box, score in zip(bboxes_nms, scores_nms):
            output.append({'category_id': cat_id, 'bbox': box, scoring: score})

    # Store proposals after NMS
    outdir = "/".join(outpath.split("/")[:-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outpath, 'w') as f:
        json.dump(output, f)


def process_all_folders(root_dir: str, scoring: str, iou_thres: float, outdir: str):
    video_src_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
    print("Doing NMS for the following dataset: {}".format(video_src_names))

    for video_src in video_src_names:
        print("Processing", video_src)
        video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root_dir, video_src, '*')))]
        video_names.sort()

        for idx, video_name in enumerate(tqdm.tqdm(video_names)):
            all_seq = glob.glob(os.path.join(root_dir, video_src, video_name, "*.json"))
            for seq in all_seq:
                json_name = seq.split("/")[-1]
                outpath = os.path.join(outdir + "_" + args.scoring, video_src, video_name, json_name)
                if args.categorywise:
                    process_one_frame_categorywise(seq, scoring, iou_thres, outpath)
                else:
                    process_one_frame(seq, scoring, iou_thres, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scoring', required=True, type=str, help='score to use during NMS')
    parser.add_argument("--scorings", required=True, nargs="+",
        help="scoring criterion to use during NMS",
    )
    parser.add_argument('--iou_thres', default=0.5, type=float, help='IoU threshold used in NMS')
    parser.add_argument('--nms_criterion', required=True, type=str, help='NMS based on bbox or mask')
    parser.add_argument('--categorywise', action='store_true', help='Only perform NMS among the same category.'
                                                                    'If not enabled, do NMS globally on every bboxes,'
                                                                    'ignoring their categories')
    parser.add_argument('--inputdir', required=True, type=str, help='input directory of orginal proposals.')
    parser.add_argument('--outdir', required=True, type=str, help='output directory of the proposals after NMS')
    args = parser.parse_args()

    for scoring in args.scorings:
        process_all_folders(args.inputdir, scoring, args.iou_thres, args.outdir)