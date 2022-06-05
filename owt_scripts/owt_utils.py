import os

import numpy as np

from typing import List

from pycocotools.mask import encode


def store_TAOnpz(predictions, input_img_path: str, valid_classes: List[int], npz_outdir: str):
    """
    Store all the proposals in one frame.
    Output in `.npz` format:

    - List of Dict:
        {"category_id": (int),
         "bbox": [x1, y1, x2, y2],
         "instance_mask": {"size": [img_h, img_w], "counts": rle_str}
         "score": (float),
         "bg_score": (float),
         "objectness": (float),
         "embeddings": (np.array), shape=(2048,)
    """
    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.npz').replace('.png', '.npz')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    npz_outpath = os.path.join(npz_outdir, frame_name)
    output = list()

    pred_classes = predictions['instances'].pred_classes

    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            bbox = predictions['instances'].pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            proposal['bbox'] = [int(b) for b in bbox]  # Convert bbox coordinates to int

            # Convert mask(numpy array) to mask(RLE)
            mask = predictions['instances'].pred_masks[i].cpu().numpy()
            mask_rle = encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            mask_rle['counts'] = mask_rle['counts'].decode(encoding="utf-8")
            proposal['instance_mask'] = mask_rle

            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolist()
            proposal['bg_score'] = predictions['instances'].bg_scores[i].cpu().numpy().tolist()
            proposal['objectness'] = predictions['instances'].objectness[i].cpu().numpy().tolist()

            embeddings = predictions['instances'].embeddings[i].cpu().numpy()
            # Each value in the shaped (2048,) embedding is between [0, 1]
            # Each value * 1e4 and store as uint16 will save a lot of storage space,
            # when using the embeddings, each value should be again divided by 1e4.
            # This encode/decode operation is equal to keeping 4 decimal digits of the original float value.
            #   uint16: Unsigned integer (0 to 65535)
            proposal['embeddings'] = (embeddings * 1e4).astype(np.uint16).tolist()

            output.append(proposal)

    np.savez_compressed(npz_outpath, output)
