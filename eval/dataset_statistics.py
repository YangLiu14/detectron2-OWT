import glob
import json
import os
import sys


def get_annotated_sequences():
    """
    Not all the sequences in TAO as annotated.
    This function is to get those annotated sequences and store as txt file.
    """

    root_dir = "/home/kloping/OpenSet_MOT/TAO_experiment/neighbour_classes_experiment/"
    fpath1 = "gt_val.json"
    with open(root_dir + fpath1, 'r') as f1:
        gt = json.load(f1)
    all_seqs = list(gt.keys())

    ArgoVerse = list()
    BDD = list()
    Charades = list()
    LaSOT = list()
    YFCC100M = list()
    for seq in all_seqs:
        data_src = seq.split("/")[0]
        if data_src == "ArgoVerse":
            ArgoVerse.append(seq.replace(".json", ".jpg"))
        elif data_src == "BDD":
            BDD.append(seq.replace(".json", ".jpg"))
        elif data_src == "Charades":
            Charades.append(seq.replace(".json", ".jpg"))
        elif data_src == "LaSOT":
            LaSOT.append(seq.replace(".json", ".jpg"))
        elif data_src == "YFCC100M":
            YFCC100M.append(seq.replace(".json", ".jpg"))
        else:
            print(seq, "ignored")

    # Write to corresponding txt files
    data_src_names = ['ArgoVerse', 'BDD', 'Charades', 'LaSOT', 'YFCC100M']
    all_list = [ArgoVerse, BDD, Charades, LaSOT, YFCC100M]
    for name, l in zip(data_src_names, all_list):
        with open('val_annotated_{}.txt'.format(name), 'w') as f:
            for item in l:
                f.write("%s\n" % item)


if __name__ == "__main__":
    # root = "/home/kloping/OpenSet_MOT/data/TAO/TAO_VAL/val/"
    # video_src_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root, '*')))]
    # print("Analysing the following dataset: {}".format(video_src_names))
    #
    # num_images = 0
    #
    # for video_src in video_src_names:
    #     video_folder_paths = glob.glob(os.path.join(root, video_src, '*'))
    #     video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root, video_src, '*')))]
    #     video_names.sort()
    #
    #     for idx, video_name in enumerate(video_names):
    #         seq = glob.glob(os.path.join(root, video_src, video_name, "*.jpg"))
    #         num_images += len(seq)
    #
    # print(num_images)

    get_annotated_sequences()