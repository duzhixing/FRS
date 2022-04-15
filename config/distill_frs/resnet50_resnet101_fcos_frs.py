_base_ = [
    '../fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'
]

model = dict(
    type='Distilling_FRS_Single',
    
    distill = dict(
        teacher_cfg='./configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py',
        teacher_model_path='./model/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.01,
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




