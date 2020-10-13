"""neighbour_classes.py
Find out the categories in TAO that can be defined as neighbour classes for each COCO-categories.
E.g. Person -> {baby, cyclist}
LHS is one of the COCO categories, and RHS are the corresponding neighbour classes"""

__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import glob
import json
import os

from typing import List, Dict


all_ids = set([i for i in range(1, 1231)])
# Category IDs in TAO that are known (appeared in COCO)
with open("../datasets/coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])

# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)



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


def load_gt(fpath: str, exclude_classes: List[str]):
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

    return gt, tao_ids


def load_proposals(folder: str, data_src: List[str], exclude_classes: List[str]):
    """
    Args:
        folder: String. The path of the folder which contains the predicted json files for each frame.
        data_src: List. Only process the videos from the data source in the list.
        exclude_classes: String.  List of classes to be ignored.

    Returns:
        props: Dict {frame_name: {bboxes, category_ids, scores}}
    """
    props = dict()
    for src in data_src:
        video_root = folder + "/" + src
        seqs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(video_root, '*')))]
        for video_seq in seqs:
            seq_dir = os.path.join(video_root, video_seq)
            frames = sorted(glob.glob(seq_dir + '/*' + '.json'))
            for frame_path in frames:
                fname_list = frame_path.split("/")
                idx = fname_list.index(src)
                fname = '/'.join(fname_list[idx:])

                # Load json
                with open(frame_path, 'r') as f:
                    frame = json.load(f)

                # TODO: map coco_id to tao_id, then store in dict
                props[fname] = frame

    return props



if __name__ == "__main__":
    # GT
    # train_annot_path = "/Users/lander14/Desktop/TAO_val_annot/annotations/train.json"
    # val_annot_path = "/Users/lander14/Desktop/TAO_val_annot/annotations/validation.json"
    val_annot_path = "/home/kloping/OpenSet_MOT/data/TAO/annotations/validation.json"

    # Pred proposals
    data_src = ['ArgoVerse_Test']
    prop_dir = "/home/kloping/OpenSet_MOT/TAO_experiment/tmp/json"

    # load_gt(val_annot_path, [])
    load_proposals(prop_dir, data_src, [])
