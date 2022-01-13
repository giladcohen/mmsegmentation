_base_ = [
    # '../_base_/models/deeplabv3_r50-d8.py',
    # '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py'
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0)
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0)
                 # paramwise_cfg=dict(
                 #     custom_keys={
                 #         'backbone': dict(lr_mult=0.0, decay_mult=0.0)}))

optimizer_config = dict()
# learning policy
lr_config = dict(policy='fixed')
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=250)
evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
find_unused_parameters = True

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHeadEmb',
        in_channels=2048,
        glove_dim=200,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.0,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DistanceLoss', loss_type='L2',
            idx_to_vec_path='/data/gilad/logs/glove_emb/pascal/glove_idx_to_emb.npy',
            class_weight=None,
            # class_weight='/data/dataset/VOCdevkit/VOC_seg_weights.npy'
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        emb=dict(emb_selection='glove', emb_path='/data/gilad/logs/glove_emb/pascal/glove_idx_to_emb.npy'),
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/single_img_memorization.txt',
        pipeline=train_pipeline),
    val=dict(
        emb=dict(emb_selection='glove', emb_path='/data/gilad/logs/glove_emb/pascal/glove_idx_to_emb.npy'),
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/single_img_memorization.txt',
        pipeline=test_pipeline),
    test=dict(
        emb=dict(emb_selection='glove', emb_path='/data/gilad/logs/glove_emb/pascal/glove_idx_to_emb.npy'),
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/single_img_memorization.txt',
        pipeline=test_pipeline))