import glob
import json
import os
import sys
import tqdm


def map_image_id2idx(annot_dict: str):
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
        res[img['id']] = i

    return res


def get_annotated_sequences2():
    root_dir = "/home/kloping/OpenSet_MOT/data/TAO/annotations/"
    fname = "validation.json"
    with open (root_dir + fname, 'r') as f:
        annot_dict = json.load(f)

    image_id2idx = map_image_id2idx(annot_dict)

    category_dict = annot_dict['categories']
    annots = annot_dict['annotations']
    tracks = annot_dict['tracks']
    videos = annot_dict['videos']
    images = annot_dict['images']


    gt_fnames = list()
    for ann in annots:
        cat_id = ann['category_id']
        idx = image_id2idx[ann['image_id']]
        img = images[idx]
        img_fname = img['file_name']
        img_fname = "/".join(img_fname.split('/')[1:])  # Get rid of "train/" or "val/" at the begining
        gt_fnames.append(img_fname)

    # Write to txt
    ArgoVerse = list()
    BDD = list()
    Charades = list()
    LaSOT = list()
    YFCC100M = list()
    for seq in gt_fnames:
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
        with open('train_annotated_{}.txt'.format(name), 'w') as f:
            for item in l:
                f.write("%s\n" % item)


def get_annotated_sequences3():
    root_dir = "/home/kloping/OpenSet_MOT/data/TAO/annotations/"
    fname = "validation.json"
    with open (root_dir + fname, 'r') as f:
        annot_dict = json.load(f)

    ArgoVerse = list()
    BDD = list()
    Charades = list()
    LaSOT = list()
    YFCC100M = list()
    AVA = list()
    HACS = list()

    for image in tqdm.tqdm(annot_dict['images']):
        img_fpath = image["file_name"]
        data_src = img_fpath.split('/')[1]
        video_name = img_fpath.split('/')[2]

        if data_src == "ArgoVerse":
            ArgoVerse.append(img_fpath)
        elif data_src == "BDD":
            BDD.append(img_fpath)
        elif data_src == "Charades":
            Charades.append(img_fpath)
        elif data_src == "LaSOT":
            LaSOT.append(img_fpath)
        elif data_src == "YFCC100M":
            YFCC100M.append(img_fpath)
        elif data_src == "AVA":
            AVA.append(img_fpath)
        elif data_src == "HACS":
            HACS.append(img_fpath)
        else:
            print(data_src + '/' + video_name, "ignored")

        ArgoVerse.sort()
        BDD.sort()
        Charades.sort()
        LaSOT.sort()
        YFCC100M.sort()
        AVA.sort()
        HACS.sort()

    data_src_names = ['ArgoVerse', 'BDD', 'Charades', 'LaSOT', 'YFCC100M', 'AVA', 'HACS']
    all_list = [ArgoVerse, BDD, Charades, LaSOT, YFCC100M, AVA, HACS]
    for name, l in zip(data_src_names, all_list):
        with open('val_annotated_{}.txt'.format(name), 'w') as f:
            for item in l:
                f.write("%s\n" % item)

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

    get_annotated_sequences3()