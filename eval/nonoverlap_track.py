#!/usr/bin/env python3

import os
import os.path as osp
import argparse
import glob
import json
import time
import tqdm
import numpy as np
from numpy import array as arr
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
import multiprocessing as mp
import logging

from pycocotools.mask import toBbox, encode, decode, area
from sklearn.metrics.pairwise import cosine_similarity
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo


# from eval.eval_keys import ObjectSize
# from eval.eval_keys import Similarity, Embedding, Distance, Tracking, Merging
# from eval.eval_keys import str_to_similarity, str_to_embedding, str_to_distance, str_to_tracking, str_to_merging
# from eval.eval_keys import similarity_to_str, embedding_to_str, distance_to_str, tracking_to_str, merging_to_str
# from eval.eval_keys import embedding_to_key
#
# from eval.eval_similarity import distance_reid
# from eval.similarity_funcs import similarity_hybrid
# from eval.eval_utils import hungarian_matching, load_proposals
from tracktor_similarity_eval import load_proposals, similarity_tracktor


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def removefile(f):
    if os.path.exists(f):
        os.remove(f)


def create_log(level=20):
    logger = logging.getLogger()
    logger.setLevel(level)
    console_handler = get_console_handler()
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger


# get logging console handler
def get_console_handler(format=None):
    console_handler = logging.StreamHandler()
    if format is None:
        formatter = logging.Formatter('%(asctime)-15s - %(levelname)s - %(message)s')
    else:
        formatter = format
    console_handler.setFormatter(formatter)
    return console_handler


# get logging file handler
def get_file_handler(file, format=None):
    file_handler = logging.FileHandler(file)
    if format is None:
        formatter = logging.Formatter('%(asctime)-15s - %(levelname)s - %(message)s')
    else:
        formatter = format
    file_handler.setFormatter(formatter)
    return file_handler


def hungarian_matching(mat, threshold=None, maximize=False):
    row_ind, col_ind = linear_sum_assignment(mat, maximize=maximize)

    if threshold is None:
        return row_ind, col_ind
    else:
        if not isinstance(threshold, float) and not isinstance(threshold, int):
            raise TypeError("Threshold value must be an integer or a float. Got {0}".format(type(threshold)))

        row_ind_thr = []
        col_ind_thr = []
        if maximize:
            mat = np.negative(mat)
            threshold = threshold * -1
        for r, c in zip(row_ind, col_ind):
            if mat[r][c] < threshold:
                row_ind_thr.append(r)
                col_ind_thr.append(c)
        return np.array(row_ind_thr), np.array(col_ind_thr)
# =======================================================================
# Tracking functions A1 -> online tracking, A2 -> online ghost tracking.
# A3 -> form short tracklets then merge them into longer tracks
# =======================================================================
def online_tracking(proposals_dir: str, opt_flow_dir: str, output_root: str, split: str, video_set: str, video: str,
                    similarity, embedding, distance, threshold: float, scoring: str,
                    use_frames_in_between: bool, ftype: str, track_id_start: int, image_dir=None, tracktor_model=None, offline=False):

    output_dir = osp.join(output_root, "tracking", "online", scoring)
    # proposals_per_video = load_proposals(proposals_dir, split, video_set, video,
    #                                                         use_frames_in_between, ftype)
    proposals_per_video = load_proposals(os.path.join(proposals_dir, video_set), video, split)
    if len(proposals_per_video.keys()) == 0:
        return
    proposals_per_video = proposals_per_video[video]

    proposals_frames = sorted(list(proposals_per_video))
    pairs = [(frame1, frame2) for frame1, frame2 in zip(proposals_frames[:-1], proposals_frames[1:])]

    all_proposals = dict()
    # for props_frame, props in proposals_per_video.items():
    #     for prop in props:
    #         all_proposals[prop['id']] = prop

    # Assign ids on the fly
    id = 0
    for props_frame, props in proposals_per_video.items():
        for prop in props:
            prop["id"] = id
            all_proposals[prop['id']] = prop
            id += 1

    tracklets = []
    new_tracklet = np.negative(np.ones(len(proposals_frames), dtype=int))

    maximize = True
    for t, (curr_frame, next_frame) in enumerate(tqdm.tqdm(pairs)):
        curr_proposals = proposals_per_video[curr_frame]
        next_proposals = proposals_per_video[next_frame]
        if len(curr_proposals) == 0:
            print("No detections in frame", t)
            continue
        curr_all_props_id = [cp['id'] for cp in curr_proposals]

        # print("Time in {0}/{1}".format(t+1, len(proposals_frames)))

        mat = np.zeros((len(curr_proposals), len(next_proposals)))
        if similarity == "tracktor":
            maximize = True
            # mat = np.array([similarity_tracktor(curr_prop, next_proposals, curr_frame, next_frame, image_dir, flow_model)
            #                 for curr_prop in curr_proposals])
            mat = np.array([similarity_tracktor(tracktor_model, curr_prop, next_proposals, curr_frame, next_frame,
                                image_dir, os.path.join(proposals_dir, video_set, video),
                                opt_flow_dir=None,
                                mode='bbox', use_frames_in_between=False) for curr_prop in curr_proposals])

        # if similarity == Similarity.bbox_iou:
        #     maximize = True
        # elif similarity == Similarity.mask_iou:
        #     maximize = True
        # elif similarity == Similarity.optical_flow:
        #     maximize = True
        # elif similarity == Similarity.reid:
        #     mat = np.array(
        #         [distance_reid(curr_prop, next_proposals, embedding_to_key(embedding), distance) for curr_prop in
        #          curr_proposals])
        #     maximize = False
        # elif similarity == Similarity.hybrid:
        #     if flow_model:
        #         mat = np.array([similarity_hybrid(curr_prop, next_proposals, curr_frame, next_frame, image_dir, flow_model)
        #                for curr_prop in curr_proposals])
        #     elif opt_flow_dir:
        #         mat = np.array([similarity_hybrid(curr_prop, next_proposals, curr_frame, next_frame, image_dir, flow_model, opt_flow_dir)
        #              for curr_prop in curr_proposals])
        #     maximize = True

        row_ind, col_ind = hungarian_matching(mat, threshold=threshold, maximize=maximize)

        curr_props_id = [curr_proposals[r_idx]['id'] for r_idx in row_ind]
        next_props_id = [next_proposals[c_idx]['id'] for c_idx in col_ind]

        if len(tracklets) == 0:  # the first tracklet is generated
            if len(curr_props_id) == 0:
                continue
            else:
                for i, curr_id in enumerate(curr_props_id):
                    nw = new_tracklet.copy()
                    nw[t] = curr_id
                    tracklets.append(nw)

        if tracklets:
            active_tracklets = np.array(tracklets)[:, t]

        if len(curr_props_id) > 0:
            for i, (curr_id, next_id) in enumerate(zip(curr_props_id, next_props_id)):
                if curr_id in active_tracklets:  # update the active tracklet
                    s = np.where(np.array(tracklets)[:, t] == curr_id)[0][0]
                    tracklets[s][t + 1] = next_id
                else:  # new tracklet
                    nw = new_tracklet.copy()
                    nw[t] = curr_id
                    nw[t + 1] = next_id
                    tracklets.append(nw)

        if len(curr_all_props_id) > 0:
            active_tracklets = np.array(tracklets)[:, t]
            for i, curr_id in enumerate(curr_all_props_id):
                if curr_id not in active_tracklets:
                    nw = new_tracklet.copy()
                    nw[t] = curr_id
                    tracklets.append(nw)

    if not offline:
        output_dir = output_dir + '/' + "tracktor"

        if use_frames_in_between:
            output_dir = output_dir + '_' + 'cfs'

        output_dir = output_dir + '_' + str(threshold)

        output_dir = osp.join(output_dir, video_set)
        mkdirs(output_dir)

        output_dir = output_dir + '/' + video + '.txt'
        removefile(output_dir)

        # Write to txt in this format
        """
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>,
        <img_h>, <img_w>, <rle>
        """
        for t_idx, tracklet in enumerate(tracklets):
            for time, id in enumerate(tracklet):
                if id == -1:
                    continue
                x1, y1, x2, y2 = all_proposals[id]['bbox']
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                conf = all_proposals[id][scoring]
                mask_rle = all_proposals[id]['instance_mask']
                img_h, img_w = mask_rle['size']
                rle_str = mask_rle['counts']
                with open(output_dir, 'a') as f:
                    line = '%d,%d,%.2f,%.2f,%.2f,%.2f,%f,-1,-1,-1,%d,%d,%s\n' % (
                        time + 1, t_idx + 1, x1, y1, w, h, conf, img_h, img_w, rle_str)
                    f.write(line)
    return tracklets, proposals_per_video, all_proposals


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

    parser = argparse.ArgumentParser(description='Evaluation Script for Non-Overlap, then Track')

    parser.add_argument('--dataset_file', type=str, default='./data/dataset_tumserver.json',
                        help='Dataset file for dataset directory and sequences list')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--video_set', type=str, required=True)
    parser.add_argument('--similarity', required=True,
                        choices=['bbox-iou', 'mask-iox', 'opt-flow', 'kalman-filter', 'tracktor', 'reid', 'hybrid'])
    parser.add_argument('--scoring', choices=['objectness', 'score', 'one_minus_bg_score',
                                              'bg_score', 'bg_rpn_sum', 'bg_rpn_product'])
    parser.add_argument('--embedding', type=str, default='maskrcnn', choices=['maskrcnn', 'premvos'])
    parser.add_argument('--offline_embedding', type=str, default='maskrcnn', choices=['maskrcnn', 'premvos'])
    parser.add_argument('--distance', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    parser.add_argument('--tracking', type=str, default='online', choices=['online', 'online-ghost', 'offline'])
    parser.add_argument('--merging', type=str, default='online', choices=['online', 'online-ghost'])
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--n_frames', type=int, default=1)
    parser.add_argument('--ftype', choices=["npz", "json"], help="File type of the proposals that are stored in.")
    parser.add_argument('--use_frames_in_between', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n_processes', type=int, default=8)
    parser.add_argument('--output_root', type=str)
    parser.add_argument('--start_vidx', type=int)
    parser.add_argument('--end_vidx', type=int)


    # Detectron2 related args
    parser.add_argument("--config-file", default="configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml",
        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    # Set up model
    ALL_START = time.time()
    print("Start counting time. ALL_START")
    # mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(opt))

    cfg = setup_cfg(opt)
    predictor = VisualizationDemo(cfg)
    # END of setting up model

    logger = create_log()
    logger.info("Logging is launched...")

    dataset_file = opt.dataset_file
    video_set = opt.video_set
    split = opt.split

    tracking = opt.tracking
    merging = None
    offline_embedding = None

    similarity = opt.similarity
    embedding = None
    distance = None
    threshold = opt.threshold
    n_frames = opt.n_frames

    use_frames_in_between = opt.use_frames_in_between
    test = opt.test

    n_processes = opt.n_processes

    with open(dataset_file) as f:
        dataset_info = json.load(f)

    dataset_root = dataset_info['dataset_root']
    # proposals_root = osp.join(dataset_info['proposals_root'], "MaskRCNN")
    proposals_root = dataset_info['proposals_root']
    opt_flow_root = dataset_info['optical_flow_root']
    proposals_dir = proposals_root

    if opt_flow_root != "":
        opt_flow_dir = osp.join(opt_flow_root, opt.video_set)
    else:
        opt_flow_dir = None
    if opt.similarity == "opt-flow" or opt.similarity == "hybrid":
        if not opt_flow_dir:
            from optical_flow_net import models
            # Set up model
            t = time.time()
            print('Setting up model')
            pwc_model_fn = 'optical_flow_net/saved_model/pwc_net.pth.tar'
            net = models.pwc_dc_net(pwc_model_fn)
            net = net.cuda()
            net.eval()
            print('Model setup, in', time.time() - t, 'seconds')
        else:
            net = None

    image_dir = osp.join(dataset_info['image_root'], opt.split, video_set)
    output_root = opt.output_root

    videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(image_dir, '*')))]

    pool = mp.Pool(n_processes)
    logger.info("Number of processes {0}.".format(pool._processes))
    """
    proposals_dir: str, opt_flow_dir: str, output_root: str, split: str, video_set: str, video: str,
    similarity: Similarity.reid,
    embedding: Embedding, distance: Distance, threshold: float, use_frames_in_between: bool,
    test: bool
    """
    # for v_idx, video in enumerate(videos):
    #     logger.info("-" * 50)
    #     logger.info("{0}/{1} Generating tracklets for {2}/{3} set".format(v_idx + 1, len(videos), video, video_set))
    #     if tracking == Tracking.online:
    #         pool.apply_async(online_tracking, args=(
    #             proposals_dir, opt_flow_dir, output_root, split, video_set, video, similarity, embedding, distance,
    #             threshold, use_frames_in_between, test))
    #     elif tracking == Tracking.online_ghost:
    #         pool.apply_async(online_tracking_ghost, args=(
    #             proposals_dir, opt_flow_dir, output_root, split, video_set, video, similarity, embedding, distance,
    #             threshold, use_frames_in_between, test, n_frames))
    #     logger.info("{0}/{1} Finished tracklets for {2}/{3} set".format(v_idx + 1, len(videos), video, video_set))
    #     logger.info("-" * 50)
    #
    # pool.close()
    # pool.join()

    """
    Without multi-processing... 
    """
    if not opt.end_vidx:
      opt.end_vidx = len(videos)
    for v_idx, video in enumerate(videos[opt.start_vidx: opt.end_vidx]):
        logger.info("-" * 50)
        logger.info("{0}/{1} Generating tracklets for {2}/{3} set".format(opt.start_vidx + v_idx + 1, len(videos), video, video_set))
        if tracking == "online":
            online_tracking(proposals_dir, "", output_root, split, video_set, video,
                            similarity, embedding, distance, threshold=threshold, scoring=opt.scoring,
                            use_frames_in_between=use_frames_in_between, ftype=opt.ftype,
                            track_id_start=v_idx*1000, image_dir=osp.join(image_dir, video), tracktor_model=predictor)
        logger.info("{0}/{1} Finished tracklets for {2}/{3} set".format(v_idx + 1, len(videos), video, video_set))
        logger.info("-" * 50)

    logger.info("Finished...")
