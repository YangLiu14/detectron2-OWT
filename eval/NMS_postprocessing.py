import argparse
import glob
import json
import math
import numpy as np
import os
import tqdm


# https://www.programmersought.com/article/97214443593/
def nms(bounding_boxes, confidence_score, threshold=0.5):
    """
    Args:
        bounding_boxes: List, Object candidate bounding boxes
        confidence_score: List, Confidence score of the bounding boxes
        threshold: float, IoU threshold

    Returns:
        List, bboxes that remains
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

    return picked_boxes, picked_score


def process_one_frame(seq: str, scoring: str, iou_thres: float, outpath: str):
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

        props_for_nms = {'bboxes': [prop['bbox']],
                         'scores': [curr_score]}

    output = list()
    bboxes_nms, scores_nms = nms(props_for_nms['bboxes'], props_for_nms['scores'], iou_thres)
    # Class-agnostic fashion: does not output category id
    for box, score in zip(bboxes_nms, scores_nms):
        output.append({'bbox': box, scoring: score})

    # Store proposals after NMS
    outdir = "/".join(outpath.split("/")[:-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outpath, 'w') as f:
        json.dump(output, f)


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
            bboxes_nms, scores_nms = nms(data['bboxes'], data['scores'], iou_thres)
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
                outpath = os.path.join(outdir, video_src, video_name, json_name)
                if args.categorywise:
                    process_one_frame_categorywise(seq, scoring, iou_thres, outpath)
                else:
                    process_one_frame(seq, scoring, iou_thres, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoring', required=True, type=str, help='score to use during NMS')
    parser.add_argument('--iou_thres', default=0.5, type=float, help='IoU threshold used in NMS')
    parser.add_argument('--categorywise', action='store_true', help='Only perform NMS among the same category.'
                                                                    'If not enabled, do NMS globally on every bboxes,'
                                                                    'ignoring their categories')
    parser.add_argument('--inputdir', required=True, type=str, help='input directory of orginal proposals.')
    parser.add_argument('--outdir', required=True, type=str, help='output directory of the proposals after NMS')
    args = parser.parse_args()
    
    process_all_folders(args.inputdir, args.scoring, args.iou_thres, args.outdir)