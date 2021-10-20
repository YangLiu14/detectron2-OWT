import glob
import numpy as np
import os
import sacred
import time
import torch
import tqdm
import yaml

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from sacred import Experiment

from config import get_output_dir
from eval.predictor import VisualizationDemo
from tracktor.tracker import Tracker


class ModelCfg:
    def __init__(self):
        self.opts = ["MODEL.WEIGHTS",
                     "/home/kloping/projects/openMOT/model_weights/model_final_be35db.pkl"]
        self.config_file = "/home/kloping/git-repos/detectron2-TAO-tracktor/configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"
        self.confidence_threshold = 0.0


ex = Experiment()

ex.add_config('cfgs/tracktor_local.yaml')

# # hacky workaround to load the corresponding configs and not having to hardcode paths here
# ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'cfgs/oracle_tracktor.yaml')


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

    cfg.freeze()
    return cfg


@ex.automain
# def inference(tracktor, reid, _config, _log, _run):
def inference(tracktor, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = os.path.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = os.path.join(output_dir, 'sacred_config.yaml')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")
    opt = ModelCfg()
    model_cfg = setup_cfg(opt)
    predictor = VisualizationDemo(model_cfg)

    # Reid network is not in use right now
    # # reid
    # reid_network = resnet50(pretrained=False, **reid['cnn'])
    # reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
    #                              map_location=lambda storage, loc: storage))
    # reid_network.eval()
    # if torch.cuda.is_available():
    #     reid_network.cuda()

    reid_network = "maskrcnn_embeddings"

    # tracktor
    if 'oracle' in tracktor:
        raise Exception("oracle tracker not implemented here.")
    else:
        tracker = Tracker(predictor.predictor, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    mot_accums = []

    video_root = "/home/kloping/projects/openMOT/data/TAO/frames/val/"
    props_root ="/home/kloping/projects/openMOT/Proposals/TAO_VAL/boxNMs_npz_score/"

    datasrcs = ["ArgoVerse", "AVA", "BDD", "Charades", "LaSOT", "YFCC100M", "HACS"]

    for datasrc in datasrcs:
        video_dir = os.path.join(video_root, datasrc)
        props_dir = os.path.join(props_root, datasrc)
        video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(video_dir, '*')))]
        for vname in video_names:
            img_paths = sorted(glob.glob(os.path.join(video_dir, vname, '*.jpg')))
            prop_paths = sorted(glob.glob(os.path.join(props_dir, vname, '*.npz')))

            tracker.reset()

            _log.info(f"Tracking: {datasrc}/{vname}")

            # I guess this is where I start tracking
            start = time.time()
            for i, (img_path, prop_path) in enumerate(zip(img_paths, prop_paths)):
                img = read_image(img_path, format="BGR")
                proposals = np.load(prop_path, allow_pickle=True)['arr_0'].tolist()
                curr_frame = {"img": img, "props": proposals}
                with torch.no_grad():
                    tracker.step(curr_frame)  # TODO: now modify this



            print("tbc")

    for seq in dataset:
        tracker.reset()

    print("tbc")




