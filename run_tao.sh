
# Experiments under CPU
./eval_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
              --eval-only MODEL.WEIGHTS /Users/lander14/Desktop/MasterThesis1/model_weights/model_final_2d9806.pkl  \
              MODEL.DEVICE cpu


# Gen proposals

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
python gen_tao_proposals_limited.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff+objectness002/json/ \
                         --video_src_name ArgoVerse \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl


# TEST Panoptic + class-agnostic
python gen_tao_proposals.py --config-file ../configs/Misc/noNMS/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/tmp/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/Panoptic_FPN_R101/model_final_be35db.pkl

# Mask RCNN
# class-aware
python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/tmp/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/detectron2/MaskRCNN_X101FPN/model_final_2d9806.pkl

# 10 FPS  vs  1.1s/image

# TAO_TRAIN evaluation
python eval_single_image_proposals.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/plot_output/ \
                                      --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/train.json \
                                      --evaluate_dir /storage/slurm/liuyang/TAO_eval/TAO_TRAIN_Proposals/Panoptic_Cas_R101_NMSoff+objectness/json/ \
                                      --score_func "score" \
                                      --do_not_timestamp

# ===============================================================================
# TAO_VAL evaluation: Recall(of GT-bbox) vs n_props
python eval_single_image_proposals.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/plot_output/ \
                                      --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                      --postNMS \
                                      --do_not_timestamp


# TAO_VAL evaluation: Recall(of GT-tracks) vs n_props
python eval_single_image_proposals_tracks.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/plot_output/ \
                                      --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                      --do_not_timestamp

# ==============================================================================
# NMS Post-processing
python NMS_postprocessing.py --inputdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff+objectness002/json/ \
                             --scoring "bg_rpn_product" \
                             --outdir "/storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/afterNMS/Panoptic_Cas_R101_NMSoff+objectness_bg*rpn/"

