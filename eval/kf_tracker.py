# ================================================================
# Kalman Filter - adapted from SORT
# https://github.com/abewley/sort
# ================================================================

import os
import numpy as np
import matplotlib
import glob
from filterpy.kalman import KalmanFilter

from detectron2.data.detection_utils import read_image
from eval.eval_utils import bbox_iou

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3, predictor=None, img=None):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    if predictor and img:
        # 1. get pred_box from trackers
        # 2. double check the bbox format of detections and trackers
        pred_box = trackers[0].tolist()[:4]   # [x1, y1, x2, y2]
        predictions = predictor.predictor(img, pred_box)
        regressed_bbox = predictions['instances'].pred_boxes[0].tensor.cpu().numpy().tolist()[0]
        import pdb; pdb.set_trace()
        # 3. Put regressed box back into trackers

    iou_matrix = iou_batch(detections, trackers)
    # Yang-modification: choose argmax here
    chosen_idx = np.argmax(iou_matrix)
    return np.array([[chosen_idx, 0]]), np.empty(0, dtype=int), np.empty((0, 5), dtype=int)

    # if min(iou_matrix.shape) > 0:
    #     a = (iou_matrix > iou_threshold).astype(np.int32)
    #     if a.sum(1).max() == 1 and a.sum(0).max() == 1:
    #         matched_indices = np.stack(np.where(a), axis=1)
    #     else:
    #         matched_indices = linear_assignment(-iou_matrix)
    # else:
    #     matched_indices = np.empty(shape=(0, 2))
    #
    # unmatched_detections = []
    # for d, det in enumerate(detections):
    #     if (d not in matched_indices[:, 0]):
    #         unmatched_detections.append(d)
    # unmatched_trackers = []
    # for t, trk in enumerate(trackers):
    #     if (t not in matched_indices[:, 1]):
    #         unmatched_trackers.append(t)
    #
    # # filter out matched with low IOU
    # matches = []
    # for m in matched_indices:
    #     if (iou_matrix[m[0], m[1]] < iou_threshold):
    #         unmatched_detections.append(m[0])
    #         unmatched_trackers.append(m[1])
    #     else:
    #         matches.append(m.reshape(1, 2))
    # if (len(matches) == 0):
    #     matches = np.empty((0, 2), dtype=int)
    # else:
    #     matches = np.concatenate(matches, axis=0)
    #
    # return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), predictor=None, img=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Yang-modification: only chose one box that matches the most
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold, predictor, img)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def similarity_kalman_filter(prop_L, props_R, frameL, frameR,
                             image_dir, prop_dir, use_frames_in_between=True):
    # args
    max_age = 1
    min_hits = 0
    iou_threshold = 0.3   # TODO: really consider how should this var be used and what value should be set.

    # The tracker takes [x1,y1,x2,y2,score] as input.
    # Set score to 0.
    det_L = np.append(prop_L['bbox'], 1).reshape(1, 5)

    mot_tracker = Sort(max_age=max_age,
                       min_hits=min_hits,
                       iou_threshold=iou_threshold)

    if use_frames_in_between:
        image_paths = sorted(glob.glob(image_dir + '/*' + '.jpg'))
        prop_paths = sorted(glob.glob(prop_dir + '/*' + '.npz'))
        idx_L = image_paths.index(os.path.join(image_dir, frameL + '.jpg'))
        idx_R = image_paths.index(os.path.join(image_dir, frameR + '.jpg'))
        assert idx_L < idx_R
        image_idxs = np.arange(idx_L, idx_R + 1)

        # Use the det_L in frameL to initialize mot_tracker
        det_L = mot_tracker.update(det_L)

        for idx in image_idxs[1:]:
            proposals = np.load(prop_paths[idx], allow_pickle=True)['arr_0'].tolist()
            dets_R = np.array([np.append(prop['bbox'], 1) for prop in proposals])
            trackers = mot_tracker.update(dets_R)

        # Match Trackers with the proposals in last frame
        box_tracker = trackers[0].tolist()[:4]
        bboxes_R = [prop['bbox'] for prop in props_R]
        ious = np.array([bbox_iou(box_tracker, box) for box in bboxes_R])
        chosen_idx = int(np.argmax(ious))
        top_iou = np.max(ious)
        return chosen_idx, top_iou


def similarity_tracktor_kalman_filter(predictor, prop_L, props_R, frameL, frameR,
                             image_dir, prop_dir, use_frames_in_between=True):
    # args
    max_age = 1
    min_hits = 0
    iou_threshold = 0.3   # TODO: really consider how should this var be used and what value should be set.

    # The tracker takes [x1,y1,x2,y2,score] as input.
    # Set score to 0.
    det_L = np.append(prop_L['bbox'], 1).reshape(1, 5)

    mot_tracker = Sort(max_age=max_age,
                       min_hits=min_hits,
                       iou_threshold=iou_threshold)

    if use_frames_in_between:
        image_paths = sorted(glob.glob(image_dir + '/*' + '.jpg'))
        prop_paths = sorted(glob.glob(prop_dir + '/*' + '.npz'))
        idx_L = image_paths.index(os.path.join(image_dir, frameL + '.jpg'))
        idx_R = image_paths.index(os.path.join(image_dir, frameR + '.jpg'))
        assert idx_L < idx_R
        image_idxs = np.arange(idx_L, idx_R + 1)

        # Use the det_L in frameL to initialize mot_tracker
        det_L = mot_tracker.update(det_L, None, None)

        for idx in image_idxs[1:]:
            img_path = image_paths[idx]
            img = read_image(img_path, format="BGR")
            proposals = np.load(prop_paths[idx], allow_pickle=True)['arr_0'].tolist()
            dets_R = np.array([np.append(prop['bbox'], 1) for prop in proposals])
            trackers = mot_tracker.update(dets_R, predictor, img)

        # Match Trackers with the proposals in last frame
        box_tracker = trackers[0].tolist()[:4]
        bboxes_R = [prop['bbox'] for prop in props_R]
        ious = np.array([bbox_iou(box_tracker, box) for box in bboxes_R])
        chosen_idx = int(np.argmax(ious))
        top_iou = np.max(ious)
        return chosen_idx, top_iou