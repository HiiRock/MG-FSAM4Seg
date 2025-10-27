default_scope = 'mmdet'
custom_imports = dict(allow_failed_imports=False, imports=[
        'mmdet.coresam',
    ])
#randomness = dict(seed=66)

env_cfg = dict( 
    cudnn_benchmark=False,  
    dist_cfg=dict(backend='nccl'),  
    mp_cfg=dict(mp_start_method='fork', 
                opencv_num_threads=0))  

hf_sam_pretrain_ckpt_path = 'work_dirs/sam_cache/sam_vit_huge/pytorch_model.bin'
hf_sam_pretrain_name = 'work_dirs/sam_cache/sam_vit_huge'

data_root = 'data/Corenew/'
dataset_type = 'CoreDataset'
num_classes = 1
num_workers = 18
crop_size = (1024,1024)  
batch_size = 1
persistent_workers = True 
indices = None 

base_lr = 0.02
max_epochs = 100
val_interval = 1
resume = False
load_from = None
work_dir = '../autodl-tmp/work_dirs/sam-lora/huge_lora-4-MHA_5fpn-last-fuse_maskrcnn'

default_hooks = dict(
    checkpoint=dict(
        interval=20,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/segm_mAP',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    #visualization=dict(type='DetVisualizationHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1, test_out_dir='vis_data')
    )



vis_backends = [dict(type='LocalVisBackend'),]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

log_level = 'INFO'  
log_processor = dict(by_epoch=True, 
                     type='LogProcessor',
                     window_size=50) 

batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            1024,
            1024,
        ),
        type='BatchFixedSizePad'),
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], 
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255], 
    bgr_to_rgb=True, 
    pad_mask=True, 
    pad_size_divisor=32, 
    batch_augments=batch_augments
)

model = dict(
    type='SAMSegMaskRCNN',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CoreSamVisionEncoder',
        extra_config=dict(output_hidden_states=True),
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(checkpoint=hf_sam_pretrain_ckpt_path,  type='Pretrained'),
        peft_config=dict(
            bias='none',
            lora_alpha=32,
            lora_dropout=0.05,
            peft_type='LORA',
            r=4,
            target_modules=['qkv']),
        ),
    neck=dict(
        type='SAMCoreFeature',
        in_channels_list='huge',
        select_layers=[32,32,32,32,32], 
        fuse = True,
        out_channels=256, 
        start_level=0,
        norm_cfg=dict(requires_grad=True, type='LN2d'),
        act_cfg=dict(type='ReLU', inplace=True),
       ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.25, 0.5, 1.0, 1.5, 2.0, ],
            scales=[8,],
            strides=[4, 8, 16, 32, 64],
            ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,),
        loss_bbox=dict(
            type='SmoothL1Loss',
            loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            ),
        bbox_head= dict(
            type='Shared2FCBBoxHead',
            in_channels=256, 
            fc_out_channels=1024, 
            roi_feat_size=7,
            num_classes = num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
            reg_class_agnostic=False, 
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),    
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
            ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict( 
                type='RoIAlign',
                output_size=14, 
                sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64], 
            ),
        mask_head=dict(
            type='FCNMaskHead',
            in_channels=256,
            num_convs=4,
            conv_out_channels=256,
            num_classes = num_classes,
            loss_mask=dict(loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            ),
        ),
    train_cfg=dict(
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                type='MaxIoUAssigner', 
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                ignore_iof_thr=-1,
                match_low_quality=True,
                ),
            sampler=dict(
                type='RandomSampler', 
                add_gt_as_proposals=False, 
                neg_pos_ub=-1,
                num=256, 
                pos_fraction=0.5, 
                ),
            debug=False,
            pos_weight=-1,
            ),
        rpn_proposal=dict( 
            nms_pre=1000, 
            max_per_img=2000, 
            min_bbox_size=0, 
            nms=dict(
                type='nms', 
                iou_threshold=0.7), 
            ),
        rcnn= dict(
            assigner=dict(
                type='MaxIoUAssigner',
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                ),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                type='RandomSampler',
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                )),
        ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            min_bbox_size=0, 
            nms=dict(type='nms', 
                     iou_threshold=0.7), 
            ),
        rcnn=dict(
            score_thr=0.05, 
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(type='nms', 
                     iou_threshold=0.5),
            ),
        ),
    )


train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip',prob=0.5),
    dict(
        type='RandomResize',
        scale=crop_size,
        keep_ratio=True,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        ),
    dict(
        type='RandomCrop',
        allow_negative_crop=True,
        crop_size=crop_size,
        crop_type='absolute', 
        recompute_bbox=True, 
        ),
    dict(
        type='FilterAnnotations', 
        by_mask=True,
        min_gt_bbox_wh=(1e-05, 1e-05),
        ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True ),
    dict(type='Resize',scale = crop_size,keep_ratio=True),
    dict(type='Pad', size = crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs', 
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),),
]

train_dataloader = dict(
    batch_size = batch_size,
    num_workers = num_workers,
    persistent_workers = True,
    sampler = dict(type='DefaultSampler',shuffle=True),
    dataset=dict(
        type = dataset_type,
        indices=None,
        data_root = data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
        )
    )

val_dataloader = dict(
    batch_size=batch_size,
    num_workers = num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler',shuffle=False,),
    dataset=dict(
        type = dataset_type,
        indices = indices,
        data_root = data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        )
    )

test_dataloader = dict(
    batch_size=batch_size,
    num_workers = num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler',shuffle=False,),
    dataset=dict(
        type = dataset_type,
        indices = indices,
        data_root = data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        )
    )

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs = max_epochs,
    val_interval = val_interval,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

find_unused_parameters = True 

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)


param_scheduler = [
    dict(type='LinearLR',begin=0, by_epoch=False, end=500, start_factor=0.001,),
    dict(
        type='MultiStepLR',
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[80, 95],
        ),
    ]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(lr=base_lr, momentum=0.9, type='SGD', weight_decay=0.0001),
    )
