import os
import shutil
import tqdm


def copy_subset_to_folder(prop_dir: str, outdir: str, datasrc: str, split='val'):
    # Load txt
    txt_path = "tao/{}_annotated_{}.txt".format(split, datasrc)
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]

    for line in tqdm.tqdm(lines):
        video_name = line.split('/')[2]
        frame_name = line.split('/')[3].replace(".jpg", ".npz")
        store_dir = os.path.join(outdir, datasrc, video_name)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        src = os.path.join(prop_dir, datasrc, video_name, frame_name)
        dst = os.path.join(store_dir, frame_name)
        shutil.copy(src, dst)


if __name__ == "__main__":
    datasrcs = ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M", "AVA", "HACS"]
    # datasrcs = ["HACS"]
    prop_dir = "/storage/user/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/preprocessed/"
    outdir = "/storage/user/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/subsets_from_preprocessed/"
    for datasrc in datasrcs:
        print("Processing", datasrc)
        copy_subset_to_folder(prop_dir, outdir, datasrc, split='val')





