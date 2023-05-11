# dataset settings
dataset_type = "TrashDataset"
data_root = "/opt/ml/dataset/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="Normalize", **img_norm_cfg),
    # Geometric Transformations
    # RandomFlip : flip_ratio(0 ~ 1), direction('vertical' 가능)
    dict(type="RandomFlip", flip_ratio=0.5, direction="horizontal"),
    # Shear : level(0 ~ 10), direction('vertical' 가능)
    dict(
        type="Shear",
        level=0,
        prob=0.5,
        direction="horizontal",
        max_shear_magnitude=0.3,
        random_negative_prob=0.5,
    ),
    # Rotate : level(0~10), max_rotate_angle(양수->시계방향)
    dict(
        type="Rotate", level=0, prob=0.5, max_rotate_angle=30, random_negative_porb=0.5
    ),
    # Translate : level(0~10), direction('vertical' 가능), min_size(tranlate 후 filtering할 최소 bbox pixel)
    dict(
        type="Translate",
        level=0,
        prob=0.5,
        direction="horizontal",
        max_translate_offset=250.0,
        random_negative_prob=0.5,
        min_size=0,
    ),
    # RandomShift : Shift_ratio(=prob), filter_thr_px(너비, 높이 threshold for filtering)
    dict(type="RandomShift", shift_ratio=0.5, max_shift_px=32, filter_thr_px=1),
    # RandomCrop : crop_size(tuple), crop_type('relative_range', 'relative', 'absolute', 'absolute_range')
    dict(
        type="RandomCrop",
        crop_size=(400, 400),
        crop_type="absolute",
        allow_negative_crop=False,
        recompute_bbox=False,
        bbox_clip_border=True,
    ),
    # Expand : mean(tuple-Dataset mean value), ratio_range(tuple-range of expand ratio), prob(float)
    dict(type="Expand", mean=(0, 0, 0), ratio_range=(1, 4), prob=0.5),
    # MinIoURandomCrop : min_ious(tuple-minimum IoU threshold for all intersections with bboxes), min_crop_size(float) -> 자세 설명 notion
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3,
        bbox_clip_border=True,
    ),
    # RandomCenterCropPad : crop_size(tuple | None - expected size after crop), ratios(tuple-random select a ratio from tuple and crop image) -> notion 설명 참조
    dict(
        type="RandomCenterCropPad",
        crop_size=None,
        ratios=(0.9, 1.0, 1.1),
        border=128,
        mean=None,
        std=None,
        to_rgb=None,
        test_mode=False,
        bbox_clip_border=True,
    ),
    # RandomAffine : randomly generates affine transform matrix(rotate, translate, shear, scale) -> notion 참조
    dict(
        type="RandomAffine",
        max_rotate_degree=10.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.5, 1.5),
        max_shear_degree=2.0,
        border=(0, 0),
        border_val=(114, 114, 114),
        min_bbox_size=2,
        min_area_ratio=0.2,
        max_aspect_ratio=20,
        bbox_clip_border=True,
        skip_filter=True,
    ),
    # Photometric Transformations
    # ColorTransform : 원본 이미지와 회색 이미지 혼합?, level(0 ~ 10)
    dict(type="ColorTransform", level=0, prob=0.5),
    # EqualizeTransform : Equalize the image histogram
    dict(type="EqualizeTransform", prob=0.5),
    # BrightnessTransform : level(0 ~ 10), 이미지 밝게 Transform
    dict(type="BrightnessTransform", level=0, prob=0.5),
    # ContrastTransform : level(0 ~ 10), 대비 조절
    dict(type="ContrastTransform", level=0, prob=0.5),
    # photoMetricDistortion : brightness_delta(int), contrast_range(tuple), hue_delta(int) -> 자세한 aug 방법은 notion 확인
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        hue_delta=18,
    ),
    # YOLOXHSVTandomAug : HSV 채널로 바꾸고 random하게 변경 후, 다시 BGR(RGB)로 바꿔주는 aug
    dict(type="YOLOXHSVRandomAug", hue_delta=5, saturation_delta=30, value_dalta=30),
    # Extra Augmentations
    # CutOut : n_holes(int | tuple(int, int)), cutout_shape(tuple(int, int) | list[tuple(int, int)]), cutout_ratio(tuple(float, float) | list[tuple(float, float)]), fill_in(tuple[int, int, int]-int에 float 가능) -> notion 참조
    dict(
        type="CutOut",
        n_holes=1,
        cutout_shape=(50, 50),
        cutout_ratio=(0.4, 0.6),
        fill_in=(0, 0, 0),
    ),
    # Mosaic : img_scale(Sequence[int]-Image size after mosaic(height, width)) -> notion 참조
    dict(
        type="Mosaic",
        img_scale=(512, 512),
        center_ratio_range=(0.5, 1.5),
        min_bbox_size=0,
        bbox_clip_border=True,
        skip_filter=True,
        pad_val=114,
        prob=1.0,
    ),
    # MixUp : img_scale(Sequence[int]-Image size after mixup(height, width)) -> notion 참조
    dict(
        type="MixUp",
        img_scale=(512, 512),
        ratio_range=(0.5, 1.5),
        flip_ratio=0.5,
        pad_val=114,
        mzx_iters=15,
        min_bbox_size=5,
        min_area_ratio=0.2,
        min_aspect_ratio=20,
        bbox_clip_border=True,
        skip_filter=True,
    ),
    # CopyPaste : Simple Copy-Paste(Strong Data Augmentation) -> notion 참조
    dict(
        type="CopyPaste",
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300,
        selected=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train.json",
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
