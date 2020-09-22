# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
from pycocotools.mask import encode, decode
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
# --input /Volumes/Elements1T/TAO_VAL/val/ --output /Users/lander14/Desktop/TAO_VAL_Proposals/viz/
# --json /Users/lander14/Desktop/TAO_VAL_Proposals/
# --opts MODEL.WEIGHTS /Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl MODEL.DEVICE cpu

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


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--json",
        help="A file or directory to save output proposals as json files. "
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    ALL_START = time.time()
    print("Start counting time. ALL_START")
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        print("Model weights", cfg.MODEL.WEIGHTS)
        # video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(args.input[0], '*'))) if fn.split('/')[-1][0] == 'b']
        video_src_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(args.input[0], '*')))]

        for video_src in video_src_names:
            video_folder_paths = glob.glob(os.path.join(args.input[0], video_src, '*'))
            video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(args.input[0], video_src, '*')))]
            video_names.sort()

            for idx, video_name in enumerate(video_names):
                print("PROCESS VIDEO {}: {}".format(idx, video_name))
                # Find all frames in the path given by args.input
                seq = glob.glob(os.path.join(args.input[0], video_src, video_name, "*.jpg"))
                # seq = glob.glob(os.path.join(args.input[0], video_name, "*.png"))
                # seq.sort()
                # seq = seq[:100]

                json_outdir = os.path.join(args.json, video_src, video_name)
                # json_outdir = args.json + video_name
                if not os.path.exists(json_outdir):
                    os.makedirs(json_outdir)

                from eval_utils import store_TAOjson
                for path in tqdm.tqdm(seq, disable=not args.output):
                    start_all = time.time()
                    # use PIL, to be consistent with evaluation
                    img = read_image(path, format="BGR")
                    # predictions, visualized_output = demo.run_on_image(img)
                    start_pred = time.time()
                    predictions = demo.predictor(img)
                    end_pred = time.time()
                    valid_classes = [i for i in range(81)]
                    store_TAOjson(predictions, path, valid_classes, json_outdir)

                    print("Inference time: {}".format(end_pred - start_pred))
                    print("All time: {}".format(time.time() - start_all))


                    # if args.output:
                    #     if not os.path.exists(args.output + "/" + video_name):
                    #         os.makedirs(args.output + "/" + video_name)
                    #     out_filename = os.path.join(args.output, video_name, os.path.basename(path))
                    #
                    #     if os.path.isdir(args.output):
                    #         assert os.path.isdir(args.output), args.output
                    #     else:
                    #         assert len(args.input) == 1, "Please specify a directory with args.output"
                    #         out_filename = args.output
                    #     visualized_output.save(out_filename)
                    # else:
                    #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    #     if cv2.waitKey(0) == 27:
                    #         break  # esc to quit

                # average_time = sum(time_list) / len(time_list)
                # # print("Average image processing time: ", str(average_time))
                #
                # json_outpath = json_outdir + "/" + "white_coco.json"
                # with open(json_outpath, 'w') as fout:
                #     json.dump(coco_output, fout)
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


