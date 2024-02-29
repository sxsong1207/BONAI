
_base_ = [
    '../_base_/models/bonai_loft_foa_r50_fpn_basic.py',
    '../_base_/schedules/schedule_2x_bonai.py', 
    '../_base_/default_runtime.py'
]

dataset_type = 'BONAI'
data_root = 'data/BONAI/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', 
         with_bbox=True,
         with_mask=True,
         with_offset=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_offsets']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
#cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
train_cities = ['shanghai', 'beijing', 'haerbin', 'chengdu']
val_cities = ['jinan']
train_ann_file = []
train_img_prefix = []
val_ann_file = []
val_img_prefix = []
for city in train_cities:
    train_ann_file.append(data_root + 'coco/bonai_{}_trainval.json'.format(city))
    train_img_prefix.append(data_root + "trainval/images/")
    
for city in val_cities:
    val_ann_file.append(data_root + 'coco/bonai_{}_trainval.json'.format(city))
    val_img_prefix.append(data_root + "trainval/images/")
    
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        bbox_type='building',
        mask_type='roof',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        gt_footprint_csv_file="",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        gt_footprint_csv_file="",
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])