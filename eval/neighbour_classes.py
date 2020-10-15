"""neighbour_classes.py
Find out the categories in TAO that can be defined as neighbour classes for each COCO-categories.
E.g. Person -> {baby, cyclist}
LHS is one of the COCO categories, and RHS are the corresponding neighbour classes"""

__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import copy
import glob
import json
import os
import tqdm
import warnings

from typing import List, Dict

# ====================================================
# Global variables
# ====================================================
all_ids = set([i for i in range(1, 1231)])
# Category IDs in TAO that are known (appeared in COCO)
with open("../datasets/coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])

# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)


def cat_id2name(tao_annot_fpath: str, coco_fpath: str):
    with open(tao_annot_fpath, 'r') as f1:
        tao_annot_dict = json.load(f1)

    with open(coco_fpath, 'r') as f2:
        coco_classes = json.load(f2)

    category_dict = tao_annot_dict['categories']
    # Mapping: category id -> category name
    tao_id2name = dict()
    for cat in category_dict:
        cat_id = cat['id']
        name = cat['name']
        tao_id2name[cat_id] = name

    coco_id2name = dict()
    for k, v in coco_classes.items():
        if k == 0:
            continue
        coco_id2name[int(k)-1] = v

    return tao_id2name, coco_id2name


# Get the cat_id --> cat_name mapping
tao_id2name, coco_id2name = cat_id2name('/storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json',
                                        '../datasets/coco/coco_classes.json')
# ====================================================
# ====================================================


def map_image_id2idx(annot_dict: str):
    """
    Map the image_id in annotatition['images'] to its index.
    Args:
        annot_dict: The annotation file (loaded from json)

    Returns:
        Dict
    """
    images = annot_dict['images']
    res = dict()
    for i, img in enumerate(images):
        res[img['id']] = i

    return res


def load_gt_and_store(fpath: str, exclude_classes: List[str]):
    """
    Args:
        fpath: The file path of the ground truth json file.
        exclude_classes: List of classes to be ignored

    Returns:
        gt: Dict, {frame_name: List[{bboxes, category_ids}]}
    """

    with open(fpath, 'r') as f:
        annot_dict = json.load(f)

    image_id2idx = map_image_id2idx(annot_dict)

    category_dict = annot_dict['categories']
    annots = annot_dict['annotations']
    tracks = annot_dict['tracks']
    videos = annot_dict['videos']
    images = annot_dict['images']


    gt = dict()
    catID2instance_count = dict()
    for ann in annots:
        cat_id = ann['category_id']
        idx = image_id2idx[ann['image_id']]
        img = images[idx]
        img_fname = img['file_name'].replace(".jpg", ".json")
        img_fname = "/".join(img_fname.split('/')[1:])  # Get rid of "train/" or "val/" at the begining
        detection = {'bbox': ann['bbox'],
                     'category_id': cat_id,
                     'track_id': ann['track_id']}

        if img_fname in gt.keys():
            gt[img_fname].append(detection)
        else:
            gt[img_fname] = [detection]

        # Get all the tao ids that actually appeared in the GT
        if cat_id in catID2instance_count.keys():
            catID2instance_count[cat_id] += 1
        else:
            catID2instance_count[cat_id] = 1
    print("Number of categories appeared in all instances:", len(catID2instance_count.keys()))

    tao_ids = list(catID2instance_count.keys())  # all tao_ids that actually appeared in the ground truth
    tao_ids.sort()

    with open('gt_val.json', 'w') as fp:
        json.dump(gt, fp)

    with open('val_tao_id.json', 'w') as f3:
        json.dump(tao_ids, f3)
    return gt, tao_ids


def load_proposals_and_store(folder: str, data_src: List[str], exclude_classes: List[str]):
    """
    Args:
        folder: String. The path of the folder which contains the predicted json files for each frame.
        data_src: List. Only process the videos from the data source in the list.
        exclude_classes: String.  List of classes to be ignored.

    Returns:
        props: Dict {frame_name: {bboxes, category_ids, scores}}
    """
    props = dict()
    print("Data source:", str(data_src))
    for src in data_src:
        print("Processing", src)
        video_root = folder + "/" + src
        seqs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(video_root, '*')))]
        for video_seq in tqdm.tqdm(seqs):
            seq_dir = os.path.join(video_root, video_seq)
            frames = sorted(glob.glob(seq_dir + '/*' + '.json'))
            for frame_path in frames:
                fname_list = frame_path.split("/")
                idx = fname_list.index(src)
                fname = '/'.join(fname_list[idx:])

                # Load json
                with open(frame_path, 'r') as f:
                    frame = json.load(f)
                proposals = list()
                # Map coco_id to tao_id, then store in dict
                # for i, p in enumerate(frame):
                #     pred_coco_id = p['category_id']
                #     if str(pred_coco_id) in coco_id2tao_id.keys():
                #         frame[i]['category_id'] = coco_id2tao_id[str(pred_coco_id)]
                #     else:
                #         frame[i]['category_id'] = -1
                for p in frame:
                    if p['score'] > 0.5:
                        proposals.append(p)
                props[fname] = proposals

    with open('val_proposals_{}.json'.format(data_src[0]), 'w') as fp:
        json.dump(props, fp)
    return props


def load_gt():
    fpath1 = "gt_val.json"
    with open(fpath1, 'r') as f1:
        gt = json.load(f1)

    fpath2 = "val_tao_id.json"
    with open(fpath2, 'r') as f2:
        tao_ids = json.load(f2)

    return gt, tao_ids


def load_proposals():
    fpath = "val_proposals_tmp.json"
    with open(fpath, 'r') as f:
        props = json.load(f)

    return props


def compute_IoU(boxA, boxB):
    """
    box in the style of (x1, y1, x2, y2)
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def find_neighbour_classes(score_thres=0.5, iou_thres=0.5):
    """
    Compare proposals(filtered with score_thres) with GT in the same frame:
        1. Calculate IoU scores, pick those IoU scores > iou_thres
        2. Construct the dict: {coco_id: [tao_id1, tao_id2, ...]} or
                               {coco_name: [tao_name1, tao_name2]}
    """

    # Load proposals and gt
    props = load_proposals()
    gt_dict, tao_ids = load_gt()

    coco2neighbour_ids = dict()
    # Find corresponding frames
    for fname, proposals in props.keys():
        try:
            gt = gt_dict[fname]
        except:
            warnings.warn(fname, "not found in GT")
            continue

        for p in proposals:
            if p['score'] < score_thres:
                continue
            p_bbox = p['bbox']
            pred_id = p['category_id']  # coco_id
            for g in gt:
                box = g['bbox']
                # convert [xc, yc, w, h] to [x1, y1, x2, y2]
                g_bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                iou = compute_IoU(p_bbox, g_bbox)
                if iou < iou_thres:
                    continue
                else:
                    # if gt_id is already in the known_tao_ids, ignore; else, add to the corresponding key
                    if g['category_id'] not in known_tao_ids:
                        if pred_id in coco2neighbour_ids.keys():
                            coco2neighbour_ids[pred_id].append(g['category_id'])
                        else:
                            coco2neighbour_ids[pred_id] = [g['category_id']]

    return coco2neighbour_ids


if __name__ == "__main__":
    # GT
    # train_annot_path = "/Users/lander14/Desktop/TAO_val_annot/annotations/train.json"
    # val_annot_path = "/Users/lander14/Desktop/TAO_val_annot/annotations/validation.json"
    # val_annot_path = "/home/kloping/OpenSet_MOT/data/TAO/annotations/validation.json"
    val_annot_path = "/storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json"

    # Pred proposals
    data_srces = [['ArgoVerse'], ['BDD'], ['Charades'], ['LaSOT'], ['YFCC100M']]
    # data_src = ['ArgoVerse']
    # prop_dir = "/home/kloping/OpenSet_MOT/TAO_experiment/tmp/json"
    prop_dir = "/storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff/json"

    # load_gt_and_store(val_annot_path, [])
    for data_src in data_srces:
        load_proposals_and_store(prop_dir, data_src, [])

    print("DONE")
