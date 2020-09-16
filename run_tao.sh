
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



# Generate proposals for TAO_VAL
python gen_tao_proposals.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
                         --input /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                         --json /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/json/ \
                         --opts MODEL.WEIGHTS /storage/slurm/liuyang/model_weights/model_final_2d9806.pkl

# TAO_VAL evaluation
python eval_single_image_proposals.py --plot_output_dir /storage/slurm/liuyang/TAO_eval/plot_output/ \
                                      --labels /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
                                      --evaluate_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/json/ \
                                      --do_not_timestamp