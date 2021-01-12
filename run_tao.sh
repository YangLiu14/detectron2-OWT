BASE_TUM=/storage/slurm/liuyang/
BASE_DAVE=/mnt/raid/davech2y/liuyang/

# Experiments under CPU
./eval_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
              --eval-only MODEL.WEIGHTS /Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl  \
              MODEL.DEVICE cpu

# =======================================
# Training
# =======================================

# ===================================
# Gen proposals
# ===================================
python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /Volumes/Elements1T/TAO_VAL/val/ --output /Users/lander14/Desktop/TAO_VAL_Proposals/viz/ \
                         --json /Users/lander14/Desktop/TAO_VAL_Proposals/ \
                         --opts MODEL.WEIGHTS /Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl MODEL.DEVICE cpu

python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /Users/lander14/Desktop/TAO_VAL/val/ --output /Users/lander14/Desktop/TAO_VAL_Proposals/viz/ \
                         --json /Users/lander14/Desktop/TAO_VAL_Proposals/ \
                         --opts MODEL.WEIGHTS /Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl MODEL.DEVICE cpu

python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --output /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/viz/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/model_final_2d9806.pkl

python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                            --input /home/kloping/OpenSet_MOT/data/TAO/TAO_VAL/val/ \
                            --json /home/kloping/OpenSet_MOT/TAO_experiment/tmp/json/ \
                            --opts MODEL.WEIGHTS /home/kloping/OpenSet_MOT/model_weights/detectron2/model_final_2d9806.pkl


# Generate proposals for TAO_VAL
python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/model_final_2d9806.pkl


# Experiment with model inference
# Class-agnostic setting with NMS on
python gen_tao_proposals.py --config-file ../configs/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSon/json/ \
                         --video_src_name ArgoVerse \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

# Class-agnostic setting with NMS off
# Panoptic + class-agnostic
python gen_tao_proposals.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff/json/ \
                         --video_src_name YFCC100M \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl


# Train Set
python gen_tao_proposals_limited.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_train/train/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_TRAIN_Proposals/Panoptic_Cas_R101_NMSoff+objectness/json/ \
                         --video_src_name YFCC100M \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

python gen_tao_proposals_limited.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_train/train/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_TRAIN_Proposals/Panoptic_Cas_R101_NMSoff+objectness/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

# Validation Set
python gen_tao_proposals.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/npz/ \
                         --video_src_name HACS \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

python gen_tao_proposals.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /mnt/raid/davech2y/liuyang/data/TAO/frames/val/ \
                         --json /mnt/raid/davech2y/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/json/ \
                         --video_src_name YFCC100M \
                         --opts MODEL.WEIGHTS /mnt/raid/davech2y/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl


python gen_tao_proposals_placeholder.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Placeholder/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

python gen_tao_proposals_subset.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/subset_experiment/Panoptic_Cas_R101_NMSoff_forTracking_Embed/json/ \
                         --video_src_name ArgoVerse  \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

# Mask RCNN
# class-aware
python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/tmp/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/MaskRCNN_X101FPN/model_final_2d9806.pkl

# 10 FPS  vs  1.1s/image

# ===============================================================================
# Evaluation: Recall vs n_props
# ===============================================================================
# TAO_VAL evaluation: Recall(of GT-bbox) vs n_props
# No NMS
python eval_recall_vs_nprops.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/plot_output_noNMSnoOverlap_gtBox/ \
                                --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff+objectness003/ \
                                --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                --recall_based_on gt_bboxes \
                                --nonOverlap \
                                --do_not_timestamp
# Post NMS
python eval_recall_vs_nprops.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/NEWplot_boxNMS_nonOverlapSmall_gtboxes/ \
                                --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/postNMS_bbox/ \
                                --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                --recall_based_on gt_bboxes \
                                --nonOverlap_small \
                                --postNMS --do_not_timestamp

python eval_recall_vs_nprops.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/NEWplot_boxNMS+nonOverlapBbox_gtboxes/ \
                                --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/postNMS_bbox/ \
                                --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                --recall_based_on gt_bboxes \
                                --nonOverlap \
                                --postNMS --do_not_timestamp


# TAO_VAL evaluation: Recall(of GT-tracks) vs n_props
# No NMS
python eval_recall_vs_nprops.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/plot_output_noNMSnoOverlap_tracks/ \
                                --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff+objectness003/ \
                                --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                --recall_based_on tracks \
                                --do_not_timestamp
# Post NMS
python eval_recall_vs_nprops.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/NEWplot_boxNMSonly_tracks/ \
                                --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/postNMS_bbox/ \
                                --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                --recall_based_on tracks \
                                --postNMS --do_not_timestamp

# Keep 1000 proposals invariant, and vary N from 1 to 100
python eval_recall_vs_NinTracks.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/NEWplot_NinTracks/ \
                                --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff+objectness003/ \
                                --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                --recall_based_on tracks \
                                --do_not_timestamp


python eval_recall_drop.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/recall_drop_experiment/ \
                           --props_base_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/postNMS_bbox/ \
                           --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                           --recall_based_on gt_bboxes \
                           --postNMS --do_not_timestamp


# ==============================================================================
# NMS Post-processing
# ==============================================================================
python NMS_postprocessing.py --scorings "bg_score" "one_minus_bg_score" "objectness" "score"  "bg_rpn_sum" "bg_rpn_product" \
                             --nms_criterion bbox \
                             --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff+objectness003/json/ \
                             --outdir "/storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/postNMS_bbox/Panoptic_Cas_R101_NMSoff+objectness003"

# TODO:
# 1. (cpu) Once the inference is done on atcremers79, run this to get postNMS proposals
# 2. (cpu) Then get nonOverlap-small for each datasrc.
# 3. (cpu) Convert dataset to mot-format and then feed them to SORT
python NMS_postprocessing.py --scorings "objectness" "bg_score" "score" "one_minus_bg_score" "bg_rpn_sum" "bg_rpn_product" \
                             --nms_criterion bbox \
                             --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/json/ \
                             --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS/

python NMS_postprocessing.py --scorings "objectness" "score" "one_minus_bg_score" \
                             --nms_criterion bbox \
                             --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/preprocessed/ \
                             --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS_npz/ \
                             --datasrc ArgoVerse

python NMS_postprocessing.py --scorings "objectness" "score" "one_minus_bg_score" \
                             --nms_criterion bbox \
                             --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/preprocessed/ \
                             --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/boxNMS_npz/ \
                             --datasrc ArgoVerse

# ==========================================================================
# Non Overlap
# ==========================================================================
python non_overlap.py --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/opt_flow_output002/ \
  --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/nonOverlap_from_optflow/ \
  --file_type .json \
  --criterion score --scoring objectness --datasrcs ArgoVerse


python non_overlap.py --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/boxNMS_npz/ \
  --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/nonOverlap_from_boxNMS/ \
  --criterion score --scoring objectness --datasrcs ArgoVerse

