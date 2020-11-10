#!/usr/bin/env python3

import cv2
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import datetime
import os
import tqdm
import re
import time

from collections import Counter
from pycocotools.mask import toBbox, encode, decode, area
from sklearn import metrics
from typing import List, Dict
from eval.eval_utils import image_stitching

# =======================================================
# Global variables
# =======================================================
"""
known_tao_ids: set of tao ids that can be mapped exactly to coco ids.
neighbor_classes: tao classes that are similar to coco_classes.
unknown_tao_ids: all_tao_ids that exclude known_tao_ids and neighbor_classes..
"""

IOU_THRESHOLD = 0.5

all_ids = set([i for i in range(1, 1231)])

# Category IDs in TAO that are known (appeared in COCO)
with open("../datasets/coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])
# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)

# neighbor classes
with open("../datasets/neighbor_classes.json") as f:
    coco2neighbor_classes = json.load(f)
# Gather tao_ids that can be categorized in the neighbor_classes
neighbor_classes = set()
for coco_id, neighbor_ids in coco2neighbor_classes.items():
    neighbor_classes = neighbor_classes.union(set(neighbor_ids))

# Exclude neighbor classes from unknown_tao_ids
unknown_tao_ids = unknown_tao_ids.difference(neighbor_classes)

# =======================================================
# =======================================================


def score_func(prop):
    if FLAGS.score_func == "score":
        return prop["score"]
    if FLAGS.score_func == "bgScore":
        return prop["bg_score"]
    if FLAGS.score_func == "1-bgScore":
        if FLAGS.postNMS:
            return prop['one_minus_bg_score']
        else:
            return 1 - prop["bg_score"]
    if FLAGS.score_func == "objectness":
        return prop["objectness"]
    if FLAGS.score_func == "bg+rpn":
        if FLAGS.postNMS:
            return prop['bg_rpn_sum']
        else:
            return (1000 * prop["objectness"] + prop["bg_score"]) / 2
    if FLAGS.score_func == "bg*rpn":
        if FLAGS.postNMS:
            return prop['bg_rpn_product']
        else:
            return math.sqrt(1000 * prop["objectness"] * prop["bg_score"])


# ==========================================================================================
# Loading functions for GT and Proposals
# ==========================================================================================
def load_gt(exclude_classes=(), ignored_sequences=(), prefix_dir_name='oxford_labels',
            dist_thresh=1000.0, area_thresh=10 * 10):
    with open(prefix_dir_name, 'r') as f:
        gt_json = json.load(f)

    # gt = { 'sequence_name': [list of bboxes(tuple)] }
    # n_boxes

    videos = gt_json['videos']
    annotations = gt_json['annotations']
    tracks = gt_json['tracks']
    images = gt_json['images']
    info = gt_json['info']
    categories = gt_json['categories']

    gt = {}
    imgID2fname = dict()
    for img in images:
        imgID2fname[img['id']] = img['file_name']

    nbox_ArgoVerse, nbox_BDD, nbox_Charades, nbox_LaSOT, nbox_YFCC100M = 0, 0, 0, 0, 0

    for ann in annotations:
        if ann["category_id"] in exclude_classes:
            continue

        img_id = ann['image_id']
        fname = imgID2fname[img_id]
        fname = fname.replace("jpg", "json")

        # ignore certain data souces
        src_name = fname.split("/")[1]
        if src_name in ignored_sequences:
            continue
        # Count n_gt_boxes in each sub-dataset
        if src_name == 'ArgoVerse':
            nbox_ArgoVerse += 1
        elif src_name == 'BDD':
            nbox_BDD += 1
        elif src_name == 'Charades':
            nbox_Charades += 1
        elif src_name == 'LaSOT':
            nbox_LaSOT += 1
        elif src_name == 'YFCC100M':
            nbox_YFCC100M += 1

        xc, yc, w, h = ann['bbox']
        # convert [xc, yc, w, h] to [x1, y1, x2, y2]
        box = (xc, yc, w, h)
        bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        if fname not in gt.keys():
            gt[fname] = {'bboxes': [], 'track_ids': [], 'category_ids': []}
        gt[fname]['bboxes'].append(bbox)
        gt[fname]['track_ids'].append(ann['track_id'])
        gt[fname]['category_ids'].append(ann['category_id'])

    n_boxes = sum([len(x['bboxes']) for x in gt.values()], 0)
    assert n_boxes == nbox_ArgoVerse + nbox_BDD + nbox_Charades + nbox_LaSOT + nbox_YFCC100M
    print("number of gt boxes", n_boxes)
    nbox = {"ArgoVerse": nbox_ArgoVerse,
            "BDD": nbox_BDD,
            "Charades": nbox_Charades,
            "LaSOT": nbox_LaSOT,
            "YFCC100M": nbox_YFCC100M}
    return gt, n_boxes, nbox


def load_gt_categories(exclude_classes=(), ignored_sequences=(), prefix_dir_name='oxford_labels'):
    gt_jsons = glob.glob("%s/*/*.json" % prefix_dir_name)

    gt_cat = {}
    gt_cat_map = {}

    cat_idx = 0
    for gt_json in gt_jsons:

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in gt_json]
        if len(matching) > 0: continue

        anns = json.load(open(gt_json))

        categories = []
        for ann in anns:
            cat_str = ann["category"]
            if cat_str in exclude_classes:
                continue
            categories.append(cat_str)

            if cat_str not in gt_cat_map:
                gt_cat_map[cat_str] = cat_idx
                cat_idx += 1

        gt_cat[gt_json] = categories
    n_boxes = sum([len(x) for x in gt_cat.values()], 0)
    print("number of gt boxes", n_boxes)
    return gt_cat, n_boxes, gt_cat_map


def load_proposals(folder, gt, ignored_sequences=(), score_fnc=score_func):
    proposals = {}
    for filename in tqdm.tqdm(gt.keys()):

        if filename.split('/')[-3] != folder.split('/')[-1]:
            continue
        prop_filename = os.path.join(folder, "/".join(filename.split("/")[-2:]))

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in filename]
        if len(matching) > 0:
            continue

        # Load proposals
        # try:
        #     props = json.load(open(prop_filename))
        # except ValueError:
        #     print("Error loading json: %s" % prop_filename)
        #     quit()
        try:
            props = json.load(open(prop_filename))
        except:
            print(prop_filename, "not found")
            continue

        if props is None:
            continue

        props = sorted(props, key=score_fnc, reverse=True)
        if FLAGS.nonOverlap:
            props = remove_mask_overlap(props)
        if FLAGS.nonOverlap_small:
            props = remove_mask_overlap_small_on_top(props)

        if "bbox" in props[0]:
            bboxes = [prop["bbox"] for prop in props]
        else:
            bboxes = [toBbox(prop["instance_mask"]) for prop in props]
            # The output box from cocoapi.mask.toBbox gives [xc, yc, w, h]
            for i in range(len(bboxes)):
                box = bboxes[i]  # in the form of [xc, yc, w, h]
                # convert [xc, yc, w, h] to [x1, y1, x2, y2]
                bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                bboxes[i] = bbox

        # convert from [x0, y0, w, h] (?) to [x0, y0, x1, y1]
        # bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]
        proposals[filename] = bboxes

    return proposals


# =========================================================================
# Non Overlap
# =========================================================================
def intersects(box1, box2):
    """
    Check if two boxes intersects with each other.
    Box are in the form of [x1, y1, x2, y2]
    Returns:
        bool
    """
    max_x1 = max(box1[0], box2[0])
    max_y1 = max(box1[1], box2[1])
    min_x2 = min(box1[2], box2[2])
    min_y2 = min(box2[3], box2[3])

    if max_x1 < min_x2 and max_y1 < min_y2:
        return True
    else:
        return False


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


def non_overlap_filter(proposals_per_frame: List[Dict]):
    """
    Filter out overlapping bboxes. If two bboxes are overlapping, keep the bbox with the higher score,
    according to the score_func.
    The bbox is already sorted according to score_fnc.

    Returns:
        remaining proposals. Same data structure as the input.
    """
    # Find out the index of the proposals to be filtered out.
    delete_idx = list()
    N = len(proposals_per_frame)
    for i in range(N):
        if i in delete_idx:
            continue
        box1 = proposals_per_frame[i]['bbox']
        for j in range(i + 1, N):
            if j in delete_idx:
                continue
            box2 = proposals_per_frame[j]['bbox']
            if intersects(box1, box2):
                delete_idx.append(j)
    delete_idx.sort()
    # print("box deleted: {} of {}".format(len(delete_idx), N))

    # Get remained proposals
    remain_props = list()
    for i in range(N):
        if delete_idx and i == delete_idx[0]:
            delete_idx.pop(0)
            continue
        else:
            remain_props.append(proposals_per_frame[i])

    return remain_props


def calculate_ious(bboxes1, bboxes2):
    """
    :param bboxes1: Kx4 matrix, assume layout (x0, y0, x1, y1)
    :param bboxes2: Nx$ matrix, assume layout (x0, y0, x1, y1)
    :return: KxN matrix of IoUs
    """
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    U = area1[:, np.newaxis] + area2[np.newaxis, :] - I
    assert (U > 0).all()
    IOUs = I / U
    assert (IOUs >= 0).all()
    assert (IOUs <= 1).all()
    return IOUs


def evaluate_proposals(gt, props, n_max_proposals=1000):
    all_ious = []  # ious for all frames
    all_track_ids = []
    all_cat_ids = []
    for img, img_gt in gt.items():
        if len(img_gt) == 0:
            continue

        try:
            img_props = props[img]
        except:
            continue

        # GT
        gt_bboxes = np.array(img_gt['bboxes'])
        track_ids = np.array(img_gt['track_ids'])
        category_ids = np.array(img_gt['category_ids'])
        # Proposals
        prop_bboxes = np.array(img_props)

        ious = calculate_ious(gt_bboxes, prop_bboxes)
        # ious = calculate_ious_category_aware(gt_bboxes, prop_bboxes, category_ids)

        # pad to n_max_proposals
        ious_padded = np.zeros((ious.shape[0], n_max_proposals))
        ious_padded[:, :ious.shape[1]] = ious[:, :n_max_proposals]
        all_ious.append(ious_padded)
        all_track_ids.append(track_ids)
        all_cat_ids.append(category_ids)
    all_ious = np.concatenate(all_ious)
    all_track_ids = np.concatenate(all_track_ids)
    all_cat_ids = np.concatenate(all_cat_ids)

    assert 0 <= IOU_THRESHOLD <= 1
    # iou_curve = [0.0 if n_max == 0 else (all_ious[:, :n_max].max(axis=1) > IOU_THRESHOLD).mean() for n_max in
    #              range(0, n_max_proposals + 1)]
    mask = (all_ious.max(axis=1) > IOU_THRESHOLD)
    recalled_cat_ids = all_cat_ids[mask]

    # Count the percentage of each category been recalled
    gt_cat2count = Counter()
    for cat_id in all_cat_ids:
        gt_cat2count[cat_id] += 1

    recalled_cat2count = Counter()
    for cat_id in recalled_cat_ids:
        recalled_cat2count[cat_id] += 1

    cat_id2ratio = dict()
    for cat_id, cnt in gt_cat2count.items():
        if cat_id in recalled_cat2count.keys():
            cat_id2ratio[cat_id] = recalled_cat2count[cat_id] / gt_cat2count[cat_id]
        else:
            cat_id2ratio[cat_id] = 0

    return cat_id2ratio


def evaluate_folder(gt, folder, ignored_sequences=(), score_fnc=score_func):
    props_postNMS = load_proposals(folder, gt, ignored_sequences=ignored_sequences, score_fnc=score_fnc)
    FLAGS.nonOverlap = True
    props_nonOverlap = load_proposals(folder, gt, ignored_sequences=ignored_sequences, score_fnc=score_fnc)
    FLAGS.nonOverlap = False
    FLAGS.nonOverlap_small = True
    props_nonOverlap_small = load_proposals(folder, gt, ignored_sequences=ignored_sequences, score_fnc=score_fnc)
    # reset
    FLAGS.nonOverlap = False
    FLAGS.nonOverlap_small = False

    cat_id2ratio_postNMS = evaluate_proposals(gt, props_postNMS)
    cat_id2ratio_nonOverlap = evaluate_proposals(gt, props_nonOverlap)
    cat_id2ratio_nonOverlap_small = evaluate_proposals(gt, props_nonOverlap_small)

    # Calculate recall drop
    # non-overlap using higher-confidence-score-mask on top
    recall_drop_nonOverlap = dict()
    for cat_id, ratio in cat_id2ratio_postNMS.items():
        if cat_id in cat_id2ratio_nonOverlap.keys():
            recall_drop_nonOverlap[str(cat_id)] = cat_id2ratio_postNMS[cat_id] - cat_id2ratio_nonOverlap[cat_id]
        else:
            recall_drop_nonOverlap[str(cat_id)] = cat_id2ratio_postNMS[cat_id] - 0

    # non-overlap using smaller-area-mask on top
    recall_drop_nonOverlap_small = dict()
    for cat_id, ratio in cat_id2ratio_postNMS.items():
        if cat_id in cat_id2ratio_nonOverlap_small.keys():
            recall_drop_nonOverlap_small[str(cat_id)] = cat_id2ratio_postNMS[cat_id] - cat_id2ratio_nonOverlap_small[cat_id]
        else:
            recall_drop_nonOverlap_small[str(cat_id)] = cat_id2ratio_postNMS[cat_id] - 0

    method_name = os.path.basename(os.path.dirname(folder + "/"))
    print("non-overlap recall drop:")
    print(recall_drop_nonOverlap)
    print("---------------------------------------------")
    print(recall_drop_nonOverlap_small)

    return recall_drop_nonOverlap, recall_drop_nonOverlap_small


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title)  # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def make_plot(export_dict, plot_title, x_vals, linewidth=5):
    plt.figure()

    itm = export_dict.items()
    itm = sorted(itm, reverse=True)
    for idx, item in enumerate(itm):
        # Compute Area Under Curve
        x = x_vals[0:1000]
        y = item[1]['data'][0:1000]
        auc = round(metrics.auc(x, y), 2)
        # curve_label = item[0].replace('.', '')
        curve_label = item[0].replace('.', '') + ': nbox(gt)=' + str(item[1]['nbox_gt']) + ', AUC=' + str(auc)
        # plt.plot(x_vals[0:700], item[1][0:700], label=curve_label, linewidth=linewidth)
        plt.plot(x_vals[0:1000], item[1]['data'][0:1000], label=curve_label, linewidth=linewidth)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xticks(np.asarray([25, 100, 200, 300, 500, 700, 900, 1000]))
    plt.xlabel("$\#$ proposals {}".format(FLAGS.score_func))
    plt.ylabel("Recall")
    ax.set_ylim([0.0, 1.0])
    plt.legend(prop={"size": 8})
    plt.grid()
    plt.title(plot_title)


def export_figs(export_dict, plot_title, output_dir, x_vals):
    # Export figs, csv
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title_to_filename(plot_title) + "_" + FLAGS.score_func + ".png"),
                    bbox_inches='tight')

        # Save to csv
        np.savetxt(os.path.join(output_dir, 'num_objects.csv'), np.array(x_vals), delimiter=',', fmt='%d')
        for item in export_dict.items():
            np.savetxt(os.path.join(output_dir, item[0] + '.csv'), item[1]['data'], delimiter=',', fmt='%1.4f')


def evaluate_all_folders_oxford(gt, plot_title, n_subset_gt_boxes, user_specified_result_dir=None, output_dir=None):
    print("----------- Evaluate Oxford Recall -----------")

    # Export dict
    export_dict = {}

    # +++ User-specified +++
    user_specified_results = None
    if user_specified_result_dir is not None:
        dirs = os.listdir(user_specified_result_dir)
        dirs.sort()

        ignore_dirs = ["HACS", "AVA"]

        all_recall_drop_nonOverlap = dict()
        all_recall_drop_nonOverlap_small = dict()
        for mydir in dirs:
            if mydir[0] == '.':
                continue  # Filter out `.DS_Store` and `._.DS_Store`
            if mydir in ignore_dirs:
                continue

            print("---Eval: %s ---" % mydir)
            results = evaluate_folder(gt, os.path.join(user_specified_result_dir, mydir))
            recall_drop_nonOverlap, recall_drop_nonOverlap_small = results

            # Add to all
            for cat_id, drop in recall_drop_nonOverlap.items():
                if str(cat_id) not in all_recall_drop_nonOverlap.keys():
                    all_recall_drop_nonOverlap[str(cat_id)] = []
                all_recall_drop_nonOverlap[str(cat_id)].append(drop)

            for cat_id, drop in recall_drop_nonOverlap_small.items():
                if str(cat_id) not in all_recall_drop_nonOverlap_small.keys():
                    all_recall_drop_nonOverlap_small[str(cat_id)] = []
                all_recall_drop_nonOverlap_small[str(cat_id)].append(drop)

            # Write to json file
            json_path = os.path.join(output_dir, plot_title, plot_title + "_" + mydir + "_" + FLAGS.score_func + ".json")
            if not os.path.exists(os.path.join(output_dir, plot_title)):
                os.makedirs(os.path.join(output_dir, plot_title))

            res = {"score_top": recall_drop_nonOverlap,
                   "area_top": recall_drop_nonOverlap_small}

            with open(json_path, 'w') as f:
                json.dump(res, f)

            print("DONE")

            # # export_dict[mydir] = user_specified_results
            # export_dict[mydir] = dict()
            # export_dict[mydir]['data'] = user_specified_results
            # export_dict[mydir]['nbox_gt'] = n_subset_gt_boxes[mydir]

        for cat_id, drop_list in all_recall_drop_nonOverlap.items():
            all_recall_drop_nonOverlap[cat_id] = sum(drop_list) / len(drop_list)
        for cat_id, drop_list in all_recall_drop_nonOverlap_small.items():
            all_recall_drop_nonOverlap_small[cat_id] = sum(drop_list) / len(drop_list)

        json_path1 = os.path.join(output_dir, plot_title, plot_title + "_allNonOverlap" + "_" + FLAGS.score_func + ".json")
        json_path2 = os.path.join(output_dir, plot_title, plot_title + "_allNonOverlapSmall" + "_" + FLAGS.score_func + ".json")
        with open(json_path1, 'w') as f:
            json.dump(all_recall_drop_nonOverlap, f)
        with open(json_path2, 'w') as f:
            json.dump(all_recall_drop_nonOverlap_small, f)

    print("DONE")


def eval_recall_oxford(output_dir):
    # ignored_seq = ("BDD", "Charades", "LaSOT", "YFCC100M", "HACS", "AVA")
    ignored_seq = ("HACS", "AVA")

    # +++ "unknown" categories +++
    print("evaluating unknown:")
    exclude_classes = tuple(known_tao_ids.union(neighbor_classes))
    gt, n_gt_boxes, n_subset_gt_boxes = load_gt(exclude_classes, ignored_seq, prefix_dir_name=FLAGS.labels)

    evaluate_all_folders_oxford(gt, "unknown",
                                n_subset_gt_boxes=n_subset_gt_boxes,
                                output_dir=output_dir,
                                user_specified_result_dir=FLAGS.evaluate_dir)

    # +++ "neighbor" categories +++
    print("evaluating neighbor classes:")
    exclude_classes = tuple(known_tao_ids.union(unknown_tao_ids))
    gt, n_gt_boxes, n_subset_gt_boxes = load_gt(exclude_classes, ignored_seq, prefix_dir_name=FLAGS.labels)

    evaluate_all_folders_oxford(gt, "COCO neighbor classes (" + str(n_gt_boxes) + " bounding boxes)",
                                n_subset_gt_boxes=n_subset_gt_boxes,
                                output_dir=output_dir,
                                user_specified_result_dir=FLAGS.evaluate_dir)

    # # +++ "known" categories +++
    # print("evaluating coco 78 classes without hot_dog and oven:")
    # exclude_classes = tuple(unknown_tao_ids.union(neighbor_classes))
    # gt, n_gt_boxes, n_subset_gt_boxes = load_gt(exclude_classes, ignored_seq, prefix_dir_name=FLAGS.labels)
    # # gt, n_gt_boxes = load_gt_oxford(exclude_classes, prefix_dir_name=FLAGS.labels)
    #
    # evaluate_all_folders_oxford(gt, "COCO known classes (" + str(n_gt_boxes) + " bounding boxes)",
    #                             n_subset_gt_boxes=n_subset_gt_boxes,
    #                             output_dir=output_dir,
    #                             user_specified_result_dir=FLAGS.evaluate_dir)


def main():
    # Matplotlib params
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams.update({'font.family': 'sans-serif'})
    matplotlib.rcParams['text.usetex'] = True

    # Prep output dir (if specified)
    output_dir = None
    if FLAGS.plot_output_dir is not None:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')

        if FLAGS.do_not_timestamp:
            timestamp = ""

        output_dir = os.path.join(FLAGS.plot_output_dir, timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eval_recall_oxford(output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument('--plot_output_dir', type=str, help='Plots output dir.')
    parser.add_argument('--props_base_dir', type=str, help='Base directory of the proposals to be evaluated.')
    parser.add_argument('--evaluate_dir', type=str, help='Dir containing result files that you want to evaluate')
    parser.add_argument('--labels', type=str, default='',
                        help='Specify dir containing the labels')
    # parser.add_argument('--labels', type=str, default='oxford_labels',
    #                     help='Specify dir containing the labels')
    parser.add_argument('--recall_based_on', required=True, type=str, help='Calculate recall based on' +
                                                                           'number of GT-bbox or on number of tracks')
    parser.add_argument('--score_func', type=str, help='Sorting criterion to use. Choose from' +
                                                       '[score, bg_score, 1-bg_score, rpn, bg+rpn, bg*rpn]')
    parser.add_argument('--postNMS', action='store_true', help='processing postNMS proposals.')
    parser.add_argument('--nonOverlap', action='store_true', help='Filter out the overlapping bboxes in proposals')
    parser.add_argument('--nonOverlap_small', action='store_true', help='Filter out the overlapping bboxes in proposals')
    parser.add_argument('--do_not_timestamp', action='store_true', help='Dont timestamp output dirs')

    FLAGS = parser.parse_args()

    # Check args
    if FLAGS.nonOverlap:
        print(">>> non-overlap (higher confidence mask on top)")
    if FLAGS.nonOverlap_small:
        print(">>> non-overlap (smaller mask on top")
    if FLAGS.recall_based_on not in ['gt_bboxes', 'tracks']:
        raise Exception(FLAGS.recall_based_on, "is not a valid option, choose from [gt_bboxes, tracks]")

    if FLAGS.postNMS:
        props_dirs = ["Panoptic_Cas_R101_NMSoff+objectness003_bg_score",
                      "Panoptic_Cas_R101_NMSoff+objectness003_bg_rpn_product",
                      "Panoptic_Cas_R101_NMSoff+objectness003_bg_rpn_sum",
                      "Panoptic_Cas_R101_NMSoff+objectness003_objectness",
                      "Panoptic_Cas_R101_NMSoff+objectness003_one_minus_bg_score",
                      "Panoptic_Cas_R101_NMSoff+objectness003_score"]
    else:
        props_dirs = ["json"]

    props_dirs = [FLAGS.props_base_dir + p for p in props_dirs]
    score_funcs = ["bgScore", "bg*rpn", "bg+rpn", "objectness", "1-bgScore", "score"]

    if FLAGS.postNMS:
        for eval_dir, score_f in zip(props_dirs, score_funcs):
            print("(postNMS) Processing", eval_dir, "using", score_f)
            FLAGS.evaluate_dir = eval_dir
            FLAGS.score_func = score_f
            main()
    else:
        for score_f in score_funcs:
            print("(normal NMS) Processing", props_dirs[0], "using", score_f)
            FLAGS.evaluate_dir = props_dirs[0]
            FLAGS.score_func = score_f
            main()
