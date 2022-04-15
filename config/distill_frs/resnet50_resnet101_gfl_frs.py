_base_ = [
    '../gfl/gfl_r50_fpn_1x_coco.py'
]

model = dict(
    type='Distilling_FRS_Single',

    distill = dict(
        teacher_cfg='./configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py',
        teacher_model_path='./model/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.1,
        distill_cls_weight=0.05,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)

custom_imports = dict(imports=['mmdet.core.utils.increase_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='Increase_Hook',)]

seed=520
#find_unused_parameters=True
