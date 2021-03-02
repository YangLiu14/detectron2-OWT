import argparse
import csv
import glob
import os
import re
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn import metrics


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title)  # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def make_plot(output_dir, export_dict, plot_title, x_vals, linewidth=5, npoints=1000):
    plt.figure()

    itm = export_dict.items()
    itm = sorted(itm, reverse=True)
    for idx, item in enumerate(itm):
        # Compute Area Under Curve
        x = x_vals[0:npoints]
        y = item[1]['data'][0:npoints]
        auc = round(metrics.auc(x, y), 2)
        curve_label = item[0].replace('.', '') + ': '  'AUC=' + str(auc)
        plt.plot(x_vals[0:npoints], item[1]['data'][0:npoints], label=curve_label, linewidth=linewidth)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    if npoints == 1000:
        ax.set_xticks(np.asarray([25, 100, 200, 300, 500, 700, 900, 1000]))
    elif npoints == 200:
        ax.set_xticks(np.asarray([25, 50, 75, 100, 125, 150, 175, 200]))
    plt.xlabel("$\#$ proposals")
    plt.ylabel("Recall")
    ax.set_ylim([0.0, 1.0])
    plt.legend(prop={"size": 8})
    plt.grid()
    plt.title(plot_title)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title_to_filename(plot_title) + ".png"),
                    bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization script for tracking result')
    parser.add_argument("--csv_folder", type=str)
    parser.add_argument("--split", type=str,
                        choices=["known", "neighbor", "unknown"])
    args = parser.parse_args()

    nboxes = {
        "known": {"all": 86405, "YFCC100M": 11088, "LaSOT": 19340, "HACS": 23522, "Charades": 8743,
                  "BDD": 7606, "ArgoVerse": 3621, "AVA": 12485},
        "neighbor": {"all": 5232, "YFCC100M": 671, "LaSOT": 1329, "HACS": 904, "Charades": 1177,
                  "BDD": 687, "ArgoVerse": 123, "AVA": 341},
        "unknown": {"all": 20522, "YFCC100M": 1690, "LaSOT": 6066, "HACS": 5462, "Charades": 4928,
                  "BDD": 28, "ArgoVerse": 116, "AVA": 2232}
    }

    split = args.split
    scoring_list = ["score",
                "bgScore",
                "1-bgScore",
                "objectness",
                "bg+objectness",
                "bg*objectness"
                ]

    total_boxes = nboxes[split]['all']

    export_dict = {"score": {},
                   "bgScore": {},
                   "1-bgScore": {},
                   "objectness": {},
                   "bg+objectness": {},
                   "bg*objectness": {}}

    print("Read in csv files")
    for scoring in tqdm.tqdm(scoring_list):
        csv_path = os.path.join(args.csv_folder, split + '_' + scoring + ".csv")
        y = list()
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                y.append(float(row[0]))

        export_dict[scoring]["data"] = y
        # if scoring == "_score":
        #     export_dict["score"]["data"] = y
        # elif scoring == "_bg_score":
        #     export_dict["bg_score"]["data"] = y
        # elif scoring == "_one_minus_bg_score":
        #     export_dict["1 - bg_score"]["data"] = y
        # elif scoring == "_objectness":
        #     export_dict["objectness"]["data"] = y
        # elif scoring == "_bg_rpn_sum":
        #     export_dict["bg + objectness"]["data"] = y
        # elif scoring == "_bg_rpn_product":
        #     export_dict["bg * objectness"]["data"] = y

    x_vals = range(1001)
    # x_vals = range(201)
    plot_title = split + " classes (" + str(total_boxes) + " bounding boxes)"

    outdir = os.path.join(args.csv_folder, "final")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    make_plot(outdir, export_dict, plot_title, x_vals, linewidth=5)


"""
python eval_recall_combine_props.py \
    --csv_folder /storage/user/liuyang/Recall_eval/tracks_only1PropInTrackIsEnough/ \
    --split unknown
"""