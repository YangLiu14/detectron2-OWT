from .mask_rcnn_R_101_FPN_100ep_LSJ import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter *= 4  # 100ep -> 400ep

lr_multiplier.scheduler.milestones = [
    milestone * 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter

# ===================================
# For OWT proposals
# disble NMS and confidence thresh
# ===================================
model.roi_heads.box_predictor.test_topk_per_image = 1000
model.roi_heads.box_predictor.test_nms_thresh = 1.0
model.roi_heads.box_predictor.test_score_thresh = 0.0
# ===================================
