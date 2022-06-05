<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

This is a forked version of Detectron2 with modifications for proposal generation as described in 
["Opening up Open-World Tracking"](https://github.com/YangLiu14/Open-World-Tracking)

To see the modifcations in detail, search globally for "OWT".
## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Model Zoo and Baselines

We use the pretrained models from [Detectron2 Model Zoo](MODEL_ZOO.md), and config them to be:
- no NMS
- no confidence threshold
- category-agnostic

We provide two examples:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>

<!-- TABLE BODY -->

<!-- ROW: panoptic_fpn_R_101_dconv_cascade_gn_3x-->
<tr><td align="left"><a href="configs/Misc/owt/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml">Panoptic FPN R101</a></td>
<td align="center">47.4</td>
<td align="center">41.3</td>
<td align="center">139797668</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl">model</a></td>
</tr>

<!-- ROW: mask_rcnn_R_101_FPN_400ep_LSJ -->
<tr><td align="left"><a href="configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ_OWT.py">R101-FPN-400ep <br/> (new baseline)</a></td>
<td align="center">48.9</td>
<td align="center">43.7</td>
<td align="center">42047764</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl">model</a></td>
</tr>

</tbody></table>

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

## Citing OWT
<pre><b>Opening up Open-World Tracking</b>
Yang Liu*, Idil Esen Zulfikar*, Jonathon Luiten*, Achal Dave*, Deva Ramanan, Bastian Leibe, Aljoša Ošep, Laura Leal-Taixé
<t><t>*Equal contribution
CVPR 2022</pre>
