_base_ = [
    # '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'ResNetV1c': dict(lr_mult=0.0)}))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
find_unused_parameters = True

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='ASPPHeadExt',
        in_channels=2048,
        glove_dim=200,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

