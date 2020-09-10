
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