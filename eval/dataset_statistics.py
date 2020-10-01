import glob
import json
import os


if __name__ == "__main__":
    root = "/home/kloping/OpenSet_MOT/data/TAO/TAO_VAL/val/"
    video_src_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root, '*')))]
    print("Analysing the following dataset: {}".format(video_src_names))

    num_images = 0

    for video_src in video_src_names:
        video_folder_paths = glob.glob(os.path.join(root, video_src, '*'))
        video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(root, video_src, '*')))]
        video_names.sort()

        for idx, video_name in enumerate(video_names):
            seq = glob.glob(os.path.join(root, video_src, video_name, "*.jpg"))
            num_images += len(seq)

    print(num_images)