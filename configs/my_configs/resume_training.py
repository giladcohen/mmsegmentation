_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
runner = dict(type='IterBasedRunner', max_iters=41000)
checkpoint_config = dict(by_epoch=False, interval=50)
evaluation = dict(interval=50, metric='mIoU', pre_eval=True)
