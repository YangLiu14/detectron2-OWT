"""statisitcal_analysis.py: Analyse the the data distribution of the TAO dataset"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import json
import matplotlib.pyplot as plt
import numpy as np
import os



all_ids = set([i for i in range(1, 1231)])

# Category IDs in TAO that are known (appeared in COCO)
with open("coco_id2tao_id.json") as f:
    coco_id2tao_id = json.load(f)
known_tao_ids = set([v for k, v in coco_id2tao_id.items()])

# Category IDs in TAO that are unknown (comparing to COCO)
unknown_tao_ids = all_ids.difference(known_tao_ids)


def long_vertical_bar_plot(height, bars, data_src):
    n = len(height)
    fig, ax = plt.subplots(figsize=(5, n // 5))  # Changing figsize depending upon data
    y_pos = np.arange(n)

    height = np.sort(height)

    ax.barh(y_pos, height, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(x) for x in bars])
    # ax.set_xlabel('Appeared in #tracks ({})'.format(data_src))
    ax.set_xlabel('#instances ({})'.format(data_src))
    ax.set_ylim(0, n)  # Manage y-axis properly

    for i, v in enumerate(height):
        ax.text(v + 3, i - 0.25, str(v), color='blue', fontweight='bold')

    # plt.show()
    plt.savefig(os.path.join('classVSnum_tracks_{}.png'.format(data_src)), dpi=300, format='png',
                bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures


def long_vertical_bar_plot_grouped(height1, height2, bars, data_src):
    n = len(height1)
    fig, ax = plt.subplots(figsize=(5, n // 5))  # Changing figsize depending upon data

    y_pos = np.arange(n)
    width = 0.35


    height1 = np.sort(height1)
    height2 = np.array(height2)


    # ax.barh(y_pos, height, align='center', color='green', ecolor='black')
    rects1 = ax.barh(y_pos - width / 2, height1, width, label='#Tracks')
    rects2 = ax.barh(y_pos + width / 2, height2, width, label='#Instances')
    # ax.barh(y_pos - width / 2, height1, color='green', ecolor='black')
    # ax.barh(y_pos + width / 2, height2, color='orange', ecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(x) for x in bars])
    # ax.set_xlabel('Appeared in #tracks ({})'.format(data_src))
    ax.set_xlabel(data_src)
    ax.set_ylim(0, n)  # Manage y-axis properly
    ax.legend()

    for i, v in enumerate(height1):
        ax.text(v + 3, i - 0.35, str(v), color='blue', fontweight='bold')

    for i, v, in enumerate(height2):
        ax.text(v + 3, i - 0.15, str(v), color='orange')

    # plt.show()
    plt.savefig(os.path.join('classVSnum_tracks_{}.png'.format(data_src)), dpi=300, format='png',
                bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures


def tao_statistics(annot_dict, data_src, top10=False):
    category_dict = annot_dict['categories']
    annots = annot_dict['annotations']
    tracks = annot_dict['tracks']

    # Mapping: category id -> category name
    catID2name = dict()
    for cat in category_dict:
        cat_id = cat['id']
        name = cat['name']
        catID2name[cat_id] = name

    # ====================================================================
    # Statistics 1:
    # For each category id, how many time it appears in different tracks.
    # ====================================================================
    cat_id2tracks1 = dict()
    for track in tracks:
        track_id = track['id']
        cat_id = track['category_id']
        if cat_id not in cat_id2tracks1.keys():
            cat_id2tracks1[cat_id] = [track_id]
        else:
            cat_id2tracks1[cat_id].append(track_id)

    print("Number of categories appeared in tracks:", len(cat_id2tracks1.keys()))

    # Sort according to k (descent)
    cat_track_ids = list()
    track_ids = list()
    for k, v in cat_id2tracks1.items():
        cat_track_ids.append((catID2name[k], k, len(v)))

    cat_track_ids.sort(key=lambda v: v[2])
    if top10:
        cat_track_ids = cat_track_ids[-10:]
        data_src = data_src + 'Top10'

    # Number of bbox appeared of each category
    catID2instance_count = dict()
    # for cat in category_dict:
    #     count = cat['instance_count']
    #     catID2instance_count[cat['id']] = count
    for ann in annots:
        cat_id = ann['category_id']
        if cat_id in catID2instance_count.keys():
            catID2instance_count[cat_id] += 1
        else:
            catID2instance_count[cat_id] = 0
    print("Number of categories appeared in all instances:", len(catID2instance_count.keys()))

    height_track = list()
    height_instances = list()
    bars = list()
    for t in cat_track_ids:
        bars.append(str(t[1]) + ': ' + t[0])
        height_track.append(t[2])
        height_instances.append(catID2instance_count[t[1]])


    cat_and_instances = list()
    for k, v in catID2instance_count.items():
        cat_and_instances.append((catID2name[k], k, v))

    cat_and_instances.sort(key=lambda v: v[2])
    if top10:
        cat_and_instances = cat_and_instances[-10:]
        data_src = data_src + 'Top10'

    height_instances_all = list()
    bars_instances = list()
    for ci in cat_and_instances:
        bars_instances.append(str(ci[1]) + ': ' + ci[0])
        height_instances_all.append(ci[2])

    # Plot:
    # long_vertical_bar_plot(height_track, bars, data_src)
    # long_vertical_bar_plot(height_instances_all, bars_instances, data_src)
    long_vertical_bar_plot_grouped(height_track, height_instances, bars, data_src)


    # # TEST
    # # From annotations
    # cat_id2tracks2 = dict()
    # for ann in annots:
    #     track_id = ann['track_id']
    #     cat_id = ann['category_id']
    #     if cat_id not in cat_id2tracks2.keys():
    #         cat_id2tracks2[cat_id] = [track_id]
    #     else:
    #         if track_id not in cat_id2tracks2[cat_id]:
    #             cat_id2tracks2[cat_id].append(track_id)
    #
    #
    # if cat_id2tracks1.keys() == cat_id2tracks2.keys():
    #     for k, v in cat_id2tracks1.items():
    #         if cat_id2tracks1[k] != cat_id2tracks2[k]:
    #             print("Nope")
    #             return False
    #     print("True")
    # else:
    #     print("False")


if __name__ == "__main__":
    train_annot_path = "/Users/lander14/Desktop/TAO_val_annot/annotations/train.json"
    with open(train_annot_path, 'r') as f:
        train_annot_dict = json.load(f)

    val_annot_path = "/Users/lander14/Desktop/TAO_val_annot/annotations/validation.json"
    with open(val_annot_path, 'r') as f:
        val_annot_dict = json.load(f)

    tao_statistics(train_annot_dict, 'train_set')
    tao_statistics(val_annot_dict, 'val_set')

    print("Done")

