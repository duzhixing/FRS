 ## Distilling Object Detectors with Feature Richness

 ####  1. Download MMdetection Framework and Dataset

Please first download [mmdetection](https://github.com/open-mmlab/mmdetection) and MS COCO2017 datasets and make sure that you can run a baseline model successfully.

 ####  2. Download the Teacher Model 

Before starting running the distillation codes, you need to download a pre-trained teacher model. We advise you to download the pretrained `faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth` and `retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth` for knowledge distillation on two-stage and one-stage students, respectively. Note that the downloading urls of the two models can be found in `mmdetection/configs/faster_rcnn/README.md` and `mmdetection/configs/retinanet/README.md`. Then, put them in the checkpoints folder as follows.

```bash
mmdetection
--model
----retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth
----faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
```

 ####  3. Change the Codes of MMdetection

1. move `distill_frs_single.py` & `distill_frs_two.py` in `mmdetection/mmdet/models/detectors/` and change `mmdetection/mmdet/models/detectors/__init__.py`

```python
from .distill_frs_single import Distilling_FRS_Single
from .distill_frs_two import Distilling_FRS_Two

# `__all__` add the follows:
__all__ = [
    'Distilling_FRS_Single', 'Distilling_FRS_Two'
]
```

2. move `adap.py` in `mmdetection/mmdet/models/necks/` and change `mmdetection/mmdet/models/necks/__init__.py`

```python
from .adap import ADAP, ADAP_C, ADAP_Residule, ADAP_SINGLE
# `__all__` add the follows:
__all__ = [
    'ADAP', 'ADAP_C', 'ADAP_Residule','ADAP_SINGLE'
]
```

3. move `distill_frs/` into `mmdetection/configs/`

4. move `increase_hook.py` in `mmdetection/mmdet/core/utils/`

 ####  4. Train model with FRS 

```bash
export CONFIG_FILE="./configs/distill_frs/resnet50_resnet101_retinanet_frs.py"
export WORK_DIR="work_dirs/retinanet_r50"
export GPU_NUM=8
bash tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    --work-dir $WORK_DIR
```

 ####  5. Test model

```bash
python tools/test.py \
    configs/retinanet/retinanet_r50_fpn_2x_coco.py \
    work_dirs/retinanet_r50/epoch_12.pth \
```