import argparse
import glob
import json
import numpy as np
import os
import tqdm

from eval.eval_utils import remove_mask_overlap, remove_mask_overlap_small_on_top


def process_one_frame(fpath:str, outdir:str, scoring: str, criterion, file_type='.json'):
    if file_type == ".json":
        with open(fpath, 'r') as f:
            proposals = json.load(f)
    elif file_type == ".npz":
        proposals = np.load(fpath, allow_pickle=True)['arr_0'].tolist()
    else:
        raise Exception("unrecognized file type.")

    if criterion == "score":
        proposals.sort(key=lambda prop: prop[scoring], reverse=True)
        processed = remove_mask_overlap(proposals)
    else:
        processed = remove_mask_overlap_small_on_top(proposals)

    # Store processed proposals
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    frame_name = fpath.split('/')[-1]
    with open(os.path.join(outdir, frame_name), 'w') as f1:
        json.dump(processed, f1)


def main(input_dir: str, outdir: str, scoring: str, criterion: str, file_type='.json'):
    videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(input_dir, '*')))]
    for idx, video in enumerate(videos):
        print("{}/{} processing {}".format(idx+1, len(videos), video))
        fpath = os.path.join(input_dir, video)
        frames = sorted(glob.glob(fpath + '/*' + file_type))
        for frame_fpath in tqdm.tqdm(frames):
            process_one_frame(frame_fpath, outdir + '/' + video, scoring, criterion, file_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", help="Input directory")
    parser.add_argument("--outdir", help="Output directory")
    parser.add_argument("--criterion", help="higher score on top or smaller area on top")
    parser.add_argument("--scoring", required=True, help="scoring criterion used tp produce the NMS result")
    parser.add_argument('--datasrcs', nargs='+', type=str, help='IoU threshold used in NMS')
    parser.add_argument('--file_type', default=".json", type=str, help='.npz or .json')
    args = parser.parse_args()

    for datasrc in args.datasrcs:
        print("Current data source:", datasrc)
        input_dir = os.path.join(args.inputdir, args.scoring, datasrc)
        if args.criterion == "score":
            outdir = os.path.join(args.outdir, "high_score_on_top", '_' + args.scoring, datasrc)
        elif args.criterion == "area":
            outdir = os.path.join(args.outdir, "small_area_on_top", '_' + args.scoring, datasrc)
        main(input_dir, outdir, args.scoring, args.criterion, file_type=args.file_type)
