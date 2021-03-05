# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
gen_tao_proposals_limited.py
Different with gen_tao_proposals.py:
here we only do inference on those annotated frames, and igore frames without annotations.
"""

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
from eval_utils import bbox_iou, compute_giou, open_flow_png_file, warp_flow, readFlow

# constants
WINDOW_NAME = "COCO detections"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
# --input /Volumes/Elements1T/TAO_VAL/val/ --output /Users/lander14/Desktop/TAO_VAL_Proposals/viz/
# --json /Users/lander14/Desktop/TAO_VAL_Proposals/
# --opts MODEL.WEIGHTS /Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl MODEL.DEVICE cpu

# ============================================================================
# Global Variable
# ============================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
known_tao_ids: set of tao ids that can be mapped exactly to coco ids.
neighbor_classes: tao classes that are similar to coco_classes.
unknown_tao_ids: all_tao_ids that exclude known_tao_ids and neighbor_classes..
"""

all_ids = set([i for i in range(1, 1231)])
# Category IDs in TAO that are known (appeared in COCO)
with open(ROOT_DIR + "/datasets/coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])
# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)
# neighbor classes
with open(ROOT_DIR + "/datasets/neighbor_classes.json") as f:
    coco2neighbor_classes = json.load(f)
# Gather tao_ids that can be categorized in the neighbor_classes
neighbor_classes = set()
for coco_id, neighbor_ids in coco2neighbor_classes.items():
    neighbor_classes = neighbor_classes.union(set(neighbor_ids))
# Exclude neighbor classes from unknown_tao_ids
unknown_tao_ids = unknown_tao_ids.difference(neighbor_classes)
# --------------------------------------------------------------------------

small_area = [0.001, 32 ^ 2]
medium_area = [32 ^ 2, 96 ^ 2]
large_area = 96 ^ 2
# ===========================================================================
# ===========================================================================

def map_image_id2fname(annot_dict: str):
    """
    Map the image_id in annotation['images'] to its index.
    Args:
        annot_dict: The annotation file (loaded from json)
    Returns:
        Dict
    """
    images = annot_dict['images']
    res = dict()
    for i, img in enumerate(images):
        res[img['id']] = img['file_name']

    return res


def load_gt(gt_path: str, datasrc: str):
    print("Loading GT")
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)

    image_id2fname = map_image_id2fname(gt_dict)

    res = dict()
    for ann in tqdm.tqdm(gt_dict['annotations']):
        cat_id = ann['category_id']
        fname = image_id2fname[ann['image_id']]
        if fname.split('/')[1] == datasrc:
            video_name = fname.split('/')[2]
            frame_name = fname.split('/')[-1].replace('.jpg', '').replace('.png', '')
            # Determine whether the current gt_obj belongs to [known, neighbor, unknown]\
            split = ''
            if cat_id in known_tao_ids:
                split = "known"
            elif cat_id in neighbor_classes:
                split = "neighbor"
            elif cat_id in unknown_tao_ids:
                split = "unknown"
            else:
                raise Exception("unrecognized category id")

            detection = {'bbox': ann['bbox'],   # [x,y,w,h]
                         'category_id': cat_id,
                         'track_id': ann['track_id'],
                         "split": split}
            if video_name not in res.keys():
                res[video_name] = dict()
            if frame_name not in res[video_name].keys():
                res[video_name][frame_name] = list()
            res[video_name][frame_name].append(detection)

    return res


def load_proposals(prop_dir, curr_video):
    datasrc = prop_dir.split('/')[-1]
    with open(ROOT_DIR + '/datasets/tao/val_annotated_{}.txt'.format(datasrc), 'r') as f:
        txt_data = f.readlines()

    print("Loading proposals in", datasrc)
    video2annot_frames = dict()
    for line in tqdm.tqdm(txt_data):
        line = line.strip()
        video_name = line.split('/')[-2]
        if video_name == curr_video:
            frame_name = line.split('/')[-1].replace(".jpg", "").replace(".png", "")
            if video_name not in video2annot_frames.keys():
                video2annot_frames[video_name] = dict()

            # Load proposals in current frame
            frame_path = os.path.join(prop_dir, video_name, frame_name + '.npz')
            proposals = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
            video2annot_frames[video_name][frame_name] = proposals

    return video2annot_frames


def match_prop_to_gt(frame_path, gt_objects):
    """
    Compare IoU of each propals in current frame with gt_objects, return the proposals that
    have highest IoU match with each gt_objects.
    If the highest IoU score < 0.5, the proposal with not be added to the returning results
    Args:
        frame_path: the file path of current frame with contains all the proposals. (.npz file)
        gt_objects: list of dict.
    Returns:
        list of proposals.
    """
    proposals = np.load(frame_path, allow_pickle=True)['arr_0'].tolist()
    # TODO: use regressed-bbox from model or use bbox converted from mask?
    # Plan1: use regressed-box directly from model
    prop_bboxes = [prop['bbox'] for prop in proposals]
    # # Plan2: use bbox converted from mask
    # prop_bboxes = [toBbox(prop['instance_mask'] for prop in proposals)]  # [x,y,w,h]
    # prop_bboxes = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in prop_bboxes]  [x1,y1,x2,y2]

    picked_props = list()
    valid_track_ids = list()
    for gt_obj in gt_objects:
        x, y, w, h = gt_obj['bbox']
        # convert [x,y,w,h] to [x1,y1,x2,y2]
        gt_box = [x, y, x+w, y+h]
        ious = np.array([bbox_iou(gt_box, box) for box in prop_bboxes])
        if np.max(ious) > 0.5:
            chosen_idx = int(np.argmax(ious))
            proposals[chosen_idx]['gt_track_id'] = gt_obj['track_id']
            proposals[chosen_idx]['split'] = gt_obj['split']

            picked_props.append(proposals[chosen_idx])
            valid_track_ids.append(gt_obj['track_id'])

    return picked_props, set(valid_track_ids)


def find_objects_in_both_frames(gt, prop_dir:str, video: str, frameL: str, frameR: str):
    """
    1. Ensure that the gt_bbox in both frames can find at least one proposals with
    IoU(gt_bbox, prop_bbox) > 0.5. Otherwise we ignore the gt_bbox.
    2. If frameL contains objects {A, B, C} and frameR contains objects {B, C, D, E},
    return objects {B, C}.
    Two objects are the same in two frames, when their `track_id` matches.
    """
    if frameL not in gt[video].keys() or frameR not in gt[video].keys():
        return [], []
    objects_L = gt[video][frameL]
    objects_R = gt[video][frameR]

    _, track_ids_L = match_prop_to_gt(os.path.join(prop_dir, video, frameL + '.npz'), objects_L)
    _, track_ids_R = match_prop_to_gt(os.path.join(prop_dir, video, frameR + '.npz'), objects_R)
    common_ids = track_ids_L.intersection(track_ids_R)

    common_objects = list()
    for obj in objects_L:
        if obj['track_id'] in common_ids:
            common_objects.append(obj)

    return common_objects, common_ids


def similarity_tracktor(predictor, prop_L, props_R, frameL, frameR,
                            image_dir, prop_dir, opt_flow_dir, mode='bbox', use_frames_in_between=True):

    bbox_L = np.array([prop_L['bbox']])
    mask_L = decode(prop_L['instance_mask'])

    image_paths = sorted(glob.glob(image_dir + '/*' + '.jpg'))
    prop_paths = sorted(glob.glob(prop_dir + '/*' + '.npz'))
    idx_L = image_paths.index(os.path.join(image_dir, frameL + '.jpg'))
    idx_R = image_paths.index(os.path.join(image_dir, frameR + '.jpg'))
    assert idx_L < idx_R

    if use_frames_in_between:
        image_idxs = np.arange(idx_L, idx_R + 1)
        if opt_flow_dir:
            flow_fpaths = sorted(glob.glob(opt_flow_dir + '/*' + '.png'))
        for idx in image_idxs[1:]:
            img_path = image_paths[idx]
            img = read_image(img_path, format="BGR")

            if opt_flow_dir:
                # warp mask_L to current frame, then do regress.
                flow_fn1 = flow_fpaths[(idx-1) * 2]
                flow_fn2 = flow_fpaths[(idx-1) * 2 + 1]

                prev_image_fn = image_paths[idx-1].replace('.jpg', '')
                prev_image_fn = prev_image_fn.split('/')[-1]
                assert prev_image_fn in flow_fn1 and prev_image_fn in flow_fn2

                flow = open_flow_png_file([flow_fn1, flow_fn2])
                warped_mask = warp_flow(mask_L, flow)  # warp flow to next frame
                warp_enc = encode(np.array(warped_mask[:, :, np.newaxis], order='F'))[0]
                warp_enc['counts'] = warp_enc['counts'].decode(encoding="utf-8")
                if area(warp_enc) > 0:
                    x, y, w, h = toBbox(warp_enc).tolist()
                    bbox_L = np.array([[x, y, x + w, y + h]])

            if mode == "kalman_filter":
                assert opt_flow_dir is None
                # TODO: to be continued

            predictions = demo.predictor(img, bbox_L)
            regressed_bbox = predictions['instances'].pred_boxes[0].tensor.cpu().numpy().tolist()[0]

            # Match the regressed bbox to current frame proposals
            proposals = np.load(prop_paths[idx], allow_pickle=True)['arr_0'].tolist()
            bboxes = [prop['bbox'] for prop in proposals]
            if mode == "bbox" or "kalman_filter":
                ious = np.array([bbox_iou(regressed_bbox, box) for box in bboxes])
            elif mode == "giou":
                ious = [compute_giou(torch.Tensor(regressed_bbox), torch.Tensor(box),
                                     bbox_inside_weights=torch.Tensor([1, 1, 1, 1]),
                                     bbox_outside_weights=torch.Tensor([1, 1, 1, 1])).item() for box in bboxes]
                ious = np.array(ious)

            matched_idx = int(np.argmax(ious))
            top_iou = np.max(ious)
            # update
            bbox_L = np.array([proposals[matched_idx]['bbox']])
            mask_L = decode(proposals[matched_idx]['instance_mask'])
        return matched_idx, top_iou

    else:
        img_path = os.path.join(image_dir, frameR + '.jpg')
        img = read_image(img_path, format="BGR")

        if opt_flow_dir:
            flow = readFlow(os.path.join(opt_flow_dir, frameL + '.flo'))
            warped_mask = warp_flow(mask_L, flow)
            warp_enc = encode(np.array(warped_mask[:, :, np.newaxis], order='F'))[0]
            warp_enc['counts'] = warp_enc['counts'].decode(encoding="utf-8")
            x, y, w, h = toBbox(warp_enc).tolist()
            bbox_L = np.array([[x, y, x + w, y + h]])

        predictions = demo.predictor(img, bbox_L)
        regressed_bbox = predictions['instances'].pred_boxes[0].tensor.cpu().numpy().tolist()[0]

        # Match the regressed bbox to the proposals in frameR
        bboxes = [prop['bbox'] for prop in props_R]
        if mode == "bbox":
            ious = np.array([bbox_iou(regressed_bbox, box) for box in bboxes])
        elif mode == "giou":
            ious = [compute_giou(torch.Tensor(regressed_bbox), torch.Tensor(box),
                                 bbox_inside_weights=torch.Tensor([1, 1, 1, 1]),
                                 bbox_outside_weights=torch.Tensor([1, 1, 1, 1])).item() for box in bboxes]
            ious = np.array(ious)

        return int(np.argmax(ious)), np.max(ious)


def eval_similarity(predictor, similarity_func: str, datasrc: str, gt_path: str, prop_dir: str, opt_flow_dir: str,
                    image_dir: str, outdir: str, pair_gap: str):
    # Only load gt and proposals relevant to current datasrc
    gt = load_gt(gt_path, datasrc)

    num_correct, num_evaled = 2422, 2707
    num_correct_known, num_evaled_known = 2341, 2609
    num_correct_neighbor, num_evaled_neighbor = 73, 80
    num_correct_unknown, num_evaled_unknown = 8, 18

    known_correct_big, known_evaled_big = 0, 0
    known_correct_medium, known_evaled_medium = 0, 0
    known_correct_small, known_evaled_small = 0, 0

    neighbor_correct_big, neighbor_evaled_big = 0, 0
    neighbor_correct_medium, neighbor_evaled_medium = 0, 0
    neighbor_correct_small, neighbor_evaled_small = 0, 0

    unknown_correct_big, unknown_evaled_big = 0, 0
    unknown_correct_medium, unknown_evaled_medium = 0, 0
    unknown_correct_small, unknown_evaled_small = 0, 0

    videos = sorted(gt.keys())
    for vidx, video in enumerate(videos[61:]):
        similarity_record = dict()
        print("{}/{} Process Videos {}/{}".format(vidx + 61, len(videos), datasrc, video))
        proposals_per_video = load_proposals(prop_dir, video)
        annot_frames = sorted(list(proposals_per_video[video]))

        if pair_gap == "1sec":
            pairs = [(frame1, frame2) for frame1, frame2 in zip(annot_frames[:-1], annot_frames[1:])]
        elif pair_gap == "5sec":
            pairs = [(frame1, frame2) for frame1, frame2 in zip(annot_frames[:-5], annot_frames[5:])]
        else:
            raise Exception("Invalid pair gap!")

        for frameL, frameR in tqdm.tqdm(pairs):
            # Record matching
            similarity_record[frameL + '|' + frameR] = dict()
            similarity_record[frameL + '|' + frameR]['matched_idxsR'] = list()

            frameL_path = os.path.join(prop_dir, video, frameL + '.npz')
            # gt_objects = gt[video][frameL]
            gt_objects, gt_track_ids = find_objects_in_both_frames(gt, prop_dir, video, frameL, frameR)
            if not gt_objects:
                continue
            similarity_record[frameL + '|' + frameR]['gt_track_ids'] = list(gt_track_ids)

            props_L, _ = match_prop_to_gt(frameL_path, gt_objects)
            props_R = np.load(os.path.join(prop_dir, video, frameR + '.npz'),
                              allow_pickle=True)['arr_0'].tolist()

            # ================================================
            # Similarity match
            # ================================================
            for propL in props_L:
                # Normal similarity functions
                if similarity_func == "tracktor-continuous-bbox":
                    matched_idx, top_iou = similarity_tracktor(predictor, propL, props_R, frameL, frameR,
                                                               os.path.join(image_dir, video),
                                                               os.path.join(prop_dir, video),
                                                               opt_flow_dir=None,
                                                               mode='bbox', use_frames_in_between=True)
                elif similarity_func == "tracktor-direct-bbox":
                    matched_idx, top_iou = similarity_tracktor(predictor, propL, props_R, frameL, frameR,
                                                               os.path.join(image_dir, video),
                                                               os.path.join(prop_dir, video),
                                                               opt_flow_dir=None,
                                                               mode='bbox', use_frames_in_between=False)
                elif similarity_func == "tracktor-continuous-optFlow":
                    matched_idx, top_iou = similarity_tracktor(predictor, propL, props_R, frameL, frameR,
                                                               os.path.join(image_dir, video),
                                                               os.path.join(prop_dir, video),
                                                               opt_flow_dir=os.path.join(opt_flow_dir, video),
                                                               mode='bbox', use_frames_in_between=True)
                elif similarity_func == "tracktor-direct-optFlow":
                    matched_idx, top_iou = similarity_tracktor(predictor, propL, props_R, frameL, frameR,
                                                               os.path.join(image_dir, video),
                                                               os.path.join(prop_dir, video),
                                                               opt_flow_dir=os.path.join(opt_flow_dir, video),
                                                               mode='bbox', use_frames_in_between=False)
                #
                # elif similarity_func == "opt-flow-mask-continuous":
                #     matched_idx, top_iou = similarity_optical_flow(propL, props_R, frameL, frameR,
                #                        os.path.join(image_dir, video),
                #                        os.path.join(prop_dir, video),
                #                        os.path.join(opt_flow_dir, video), mode='mask', use_frames_in_between=True)
                # elif similarity_func == "opt-flow-bbox-continuous":
                #     matched_idx, top_iou = similarity_optical_flow(propL, props_R, frameL, frameR,
                #                        os.path.join(image_dir, video),
                #                        os.path.join(prop_dir, video),
                #                        os.path.join(opt_flow_dir, video), mode='bbox', use_frames_in_between=True)
                #
                # elif similarity_func == "kalman-filter":
                #     matched_idx, top_iou = similarity_kalman_filter(propL, props_R, frameL, frameR,
                #                                                    os.path.join(image_dir, video),
                #                                                    os.path.join(prop_dir, video),
                #                                                    use_frames_in_between=True)

                # Compare the matched_propR with gt_objects
                propR = props_R[matched_idx]
                match = False
                gt_objects_R = gt[video][frameR]
                for obj in gt_objects_R:
                    if obj['track_id'] == propL['gt_track_id']:
                        x, y, w, h = obj['bbox']
                        gt_box = [x, y, x + w, y + h]
                        if bbox_iou(gt_box, propR['bbox']) > 0.5:
                            match = True
                            break

                if match:
                    similarity_record[frameL + '|' + frameR]['matched_idxsR'].append(matched_idx)

                if propL['split'] == "known":
                    num_correct_known += match
                    num_evaled_known += 1
                    # Statistics of area
                    x1, y1, x2, y2 = propL['bbox']
                    w, h = x2 - x1, y2 - y1
                    bbox_area = w * h
                    if small_area[0] <= bbox_area < small_area[1]:
                        known_correct_small += match
                        known_evaled_small += 1
                    elif medium_area[0] <= bbox_area < medium_area[1]:
                        known_correct_medium += match
                        known_evaled_medium += 1
                    elif large_area <= bbox_area:
                        known_correct_big += match
                        known_evaled_big += 1

                elif propL['split'] == "neighbor":
                    num_correct_neighbor += match
                    num_evaled_neighbor += 1
                    # Statistics of area
                    x1, y1, x2, y2 = propL['bbox']
                    w, h = x2 - x1, y2 - y1
                    bbox_area = w * h
                    if small_area[0] <= bbox_area < small_area[1]:
                        neighbor_correct_small += match
                        neighbor_evaled_small += 1
                    elif medium_area[0] <= bbox_area < medium_area[1]:
                        neighbor_correct_medium += match
                        neighbor_evaled_medium += 1
                    elif large_area <= bbox_area:
                        neighbor_correct_big += match
                        neighbor_evaled_big += 1

                elif propL['split'] == "unknown":
                    num_correct_unknown += match
                    num_evaled_unknown += 1
                    # Statistics of area
                    x1, y1, x2, y2 = propL['bbox']
                    w, h = x2 - x1, y2 - y1
                    bbox_area = w * h
                    if small_area[0] <= bbox_area < small_area[1]:
                        unknown_correct_small += match
                        unknown_evaled_small += 1
                    elif medium_area[0] <= bbox_area < medium_area[1]:
                        unknown_correct_medium += match
                        unknown_evaled_medium += 1
                    elif large_area <= bbox_area:
                        unknown_correct_big += match
                        unknown_evaled_big += 1

                num_correct += match
                num_evaled += 1

        print("Current accuracy:            {}/{}".format(num_correct, num_evaled))
        print("Current accuracy (known):    {}/{}".format(num_correct_known, num_evaled_known))
        print("Current accuracy (neighbor): {}/{}".format(num_correct_neighbor, num_evaled_neighbor))
        print("Current accuracy (unknown):  {}/{}".format(num_correct_unknown, num_evaled_unknown))
        print("--------------------------------------------------------------")
        print("(known) small: {}/{}; medium: {}/{}; large: {}/{}".format(known_correct_small, known_evaled_small,
            known_correct_medium, known_evaled_medium, known_correct_big, known_evaled_big))
        print("(neigh) small: {}/{}; medium: {}/{}; large: {}/{}".format(neighbor_correct_small, neighbor_evaled_small,
            neighbor_correct_medium, neighbor_evaled_medium, neighbor_correct_big, neighbor_evaled_big))
        print("(unknw) small: {}/{}; medium: {}/{}; large: {}/{}".format(unknown_correct_small, unknown_evaled_small,
            unknown_correct_medium, unknown_evaled_medium, unknown_correct_big, unknown_evaled_big))

        if outdir:
            with open(outdir + '/' + video + '.json', 'w') as fout:
                json.dump(similarity_record, fout)

    print("----------------------------------------------------------------")
    print("-------------------- Final Results -----------------------------")
    print("(known) small: {}/{}; medium: {}/{}; large: {}/{}".format(known_correct_small, known_evaled_small,
        known_correct_medium, known_evaled_medium, known_correct_big, known_evaled_big))
    print("(neigh) small: {}/{}; medium: {}/{}; large: {}/{}".format(neighbor_correct_small, neighbor_evaled_small,
        neighbor_correct_medium, neighbor_evaled_medium, neighbor_correct_big, neighbor_evaled_big))
    print("(unknw) small: {}/{}; medium: {}/{}; large: {}/{}".format(unknown_correct_small, unknown_evaled_small,
        unknown_correct_medium, unknown_evaled_medium, unknown_correct_big, unknown_evaled_big))
    print("-----------------------------------------------------------------")
    print("Top 1 accuracy =            {}/{} = {}".format(num_correct, num_evaled, num_correct/num_evaled))
    print("Top 1 accuracy (known) =    {}/{}".format(num_correct_known, num_evaled_known))
    print("Top 1 accuracy (neighbor) = {}/{}".format(num_correct_neighbor, num_evaled_neighbor))
    print("Top 1 accuracy (unknown) =  {}/{}".format(num_correct_unknown, num_evaled_unknown))

    result_outpath = os.path.join(outdir, datasrc + '_finalResult.txt')
    with open(result_outpath, 'a+') as f:
        f.write('-' * 65 + '\n')
        f.write('-' * 25 + f' Final Results ' + '-' * 25 + '\n')
        f.write(
            f'(known) small: {known_correct_small}/{known_evaled_small}; medium: {known_correct_medium}/{known_evaled_medium}; large: {known_correct_big}/{known_evaled_big} \n')
        f.write(
            f'(neigh) small: {neighbor_correct_small}/{neighbor_evaled_small}; medium: {neighbor_correct_medium}/{neighbor_evaled_medium}; large: {neighbor_correct_big}/{neighbor_evaled_big} \n')
        f.write(
            f'(unknw) small: {unknown_correct_small}/{unknown_evaled_small}; medium: {unknown_correct_medium}/{unknown_evaled_medium}; large: {unknown_correct_big}/{unknown_evaled_big} \n')

        f.write('-' * 65 + '\n')
        f.write(f'Top 1 accuracy =            {num_correct}/{num_evaled} = {num_correct / num_evaled}\n')
        f.write(f'Top 1 accuracy (known) =    {num_correct_known}/{num_evaled_known}\n')
        f.write(f'Top 1 accuracy (neighbor) = {num_correct_neighbor}/{num_evaled_neighbor}\n')
        f.write(f'Top 1 accuracy (unknown) =  {num_correct_unknown}/{num_evaled_unknown}\n')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script for Proposal Similarity')
    parser.add_argument('--datasrc', required=True, type=str, help='Current datasrc to process. '
                        '["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M", "AVA", "HACS"]')
    parser.add_argument('--image_dir', required=True, type=str, help='Root directory stores all the images')
    parser.add_argument('--prop_dir', required=True, type=str, help='Root directory stores all the proposals')
    parser.add_argument('--gt_path', required=True, type=str, help='File path to GT annotation file.')
    parser.add_argument('--opt_flow_dir', type=str, help='Root directory stores all the optical flows')
    parser.add_argument('--outdir', type=str, help='Output directory of intermediate results')
    parser.add_argument('--pair_gap', choices=['1sec', '5sec'], type=str, help='At what time gap will the pairs of '
                        'frameL, frameR be chosen.')
    parser.add_argument('--similarity_func', required=True,
                        # choices=['bbox-iou', 'mask-iox', 'opt-flow', 'kalman-filter', 'tracktor', 'reid'],
                        help='Similarity function used in this script.')
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

    datasrc = args.datasrc
    image_dir = os.path.join(args.image_dir, datasrc)
    prop_dir = os.path.join(args.prop_dir, datasrc)
    gt_path = args.gt_path
    if args.opt_flow_dir:
        opt_flow_dir = os.path.join(args.opt_flow_dir, datasrc)
    else:
        opt_flow_dir = None
    if args.outdir:
        outdir = os.path.join(args.outdir, args.similarity_func + '_' + args.pair_gap, datasrc)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir = None

    eval_similarity(demo, args.similarity_func, datasrc, gt_path, prop_dir, opt_flow_dir, image_dir, outdir, args.pair_gap)
