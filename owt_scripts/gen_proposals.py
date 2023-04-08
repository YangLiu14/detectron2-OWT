"""
This script is for generating proposals for the experiements in "Opening up Open-World Tracking"
Adapted from demo/demo.py
"""
import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import time

import tqdm
import torch

from pycocotools.mask import encode, decode

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from owt_utils import store_TAOnpz

# constants
WINDOW_NAME = "Proposal Generation for OWT"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.0
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
        "--outdir",
        help="output directory to save the output proposals as npz file.",
    )

    parser.add_argument(
        "--video_src_names",
        nargs='+',
        help="Specify the list of video_src_name (the name of the dataset) that you want to process. "
             "Otherwise, process every folder in the given root directory.",
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
    parser.add_argument(
        "--annot-only",
        help="Only generate proposals for annotated frames in TAO dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="val"
    )

    # Start and end idx of videos inside a `video_src`.
    parser.add_argument(
        "--vidx_start", default=0, type=int,
        help="start processing video from this index",
    )
    parser.add_argument(
        "--vidx_end", default=10000000000, type=int,
        help="processing video until this index",
    )
    return parser


def preprocess_image(original_image, input_format="BGR"):
    aug = T.ResizeShortestEdge([800, 800], 1333)

    # Apply pre-processing to image.
    if input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": height, "width": width}
    return inputs


def inference(model, img, config_file):
    if config_file.endswith("yaml"):
        predictions = model.predictor(img)
    elif config_file.endswith("py"):
        inputs = preprocess_image(img)
        predictions = model([inputs])[0]
    else:
        raise Exception(f"{config_file} is not supported.")

    return predictions


if __name__ == "__main__":
    ALL_START = time.time()
    print("Start counting time. ALL_START")
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if args.config_file.endswith("yaml"):
        cfg = setup_cfg(args)
        model = VisualizationDemo(cfg)
    elif args.config_file.endswith("py"):
        model = model_zoo.get(args.config_file, trained=True)
        model.cuda()
        model.eval()
    else:
        raise Exception(f"{args.config_file} is not supported.")

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(osp.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        data_dir = osp.join(args.input[0], args.split)
        if args.video_src_names:   # only process the given video sources
            video_src_names = args.video_src_names
        else:
            video_src_names = [fn.split('/')[-1] for fn in sorted(glob.glob(osp.join(data_dir, '*')))]
        print("Processing the following datasets: {}".format(video_src_names))

        for video_src in video_src_names:
            video_folder_paths = glob.glob(osp.join(data_dir, video_src, '*'))
            video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(osp.join(data_dir, video_src, '*')))]
            video_names.sort()

            if args.annot_only:
                txt_fname = f"../datasets/tao/{args.split}_annotated_{video_src}.txt"
                with open(txt_fname) as f:
                    content = f.readlines()
                seq_names_from_txt = [osp.join(args.input[0], x.strip()) for x in content]

                for path in tqdm.tqdm(seq_names_from_txt[args.vidx_start: args.vidx_end]):
                    path_split = path.split('/')
                    idx = path_split.index(video_src)
                    video_name = path_split[idx + 1]
                    curr_outdir = osp.join(args.outdir, video_src, video_name)
                    # json_outdir = args.json + video_name
                    if not os.path.exists(curr_outdir):
                        os.makedirs(curr_outdir)

                    with torch.no_grad():
                        img = read_image(path, format="BGR")
                        predictions = inference(model, img, args.config_file)

                        valid_classes = [i for i in range(81)]
                        store_TAOnpz(predictions, path, valid_classes, curr_outdir)

            else:
                for idx, video_name in enumerate(video_names[args.vidx_start: args.vidx_end]):
                    print("PROCESS VIDEO {}: {}".format(idx, video_name))
                    # Find all frames in the path given by args.input
                    seq = sorted(glob.glob(osp.join(data_dir, video_src, video_name, "*.jpg"))) + \
                          sorted(glob.glob(osp.join(data_dir, video_src, video_name, "*.png")))

                    curr_outdir = os.path.join(args.outdir, video_src, video_name)
                    if not os.path.exists(curr_outdir):
                        os.makedirs(curr_outdir)

                    for path in tqdm.tqdm(seq):
                        start_all = time.time()
                        # use PIL, to be consistent with evaluation
                        with torch.no_grad():
                            img = read_image(path, format="BGR")
                            predictions = inference(model, img, args.config_file)

                            valid_classes = [i for i in range(81)]
                            store_TAOnpz(predictions, path, valid_classes, curr_outdir)
