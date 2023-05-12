import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device
import inference
import wandb

wandb.login()

base_dir = os.getcwd() + "/custom_configs"
checkpoint_dir = os.getcwd() + "/checkpoint/custom_1x"
root = "../../dataset/"

# load config file
print(base_dir + "/custom_1x.py")
cfg = Config.fromfile(base_dir + "/custom_1x.py")

# model config 수정
cfg.model.roi_head.bbox_head.num_classes = 10
cfg.model.roi_head.bbox_head.loss_bbox = dict(type="CIoULoss", loss_weight=12.0)

# dataset config 수정
# cfg.data.train.pipeline[14]["img_scale"] = (1024, 1024)

# cfg.data.val.ann_file = root + "train.json"
# cfg.data.val.pipeline[1]["img_scale"] = (1024, 1024)

# cfg.data.test.pipeline[1]["img_scale"] = (1024, 1024)
cfg.data.test.test_mode = True

cfg.data.samples_per_gpu = 4

# optimizer config 수정
# cfg.optimizer = dict(type="Adam", lr=0.0002, weight_decay=0.001)
cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

# 기타 config 수정
cfg.seed = 2023
cfg.gpu_ids = [0]
cfg.work_dir = checkpoint_dir
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()
cfg.runner.max_epochs = 1
cfg.evaluation.classwise = True

# 최종 config 저장
os.makedirs(checkpoint_dir, exist_ok=True)
with open(checkpoint_dir + "/exp.py", "w") as f:
    f.write(cfg.pretty_text)

# wandb config 설정
cfg.log_config.hooks = [
    dict(type="TextLoggerHook"),
    dict(
        type="MMDetWandbHook",
        init_kwargs={
            "project": "(model_name)",
            "name": "(backbone)_(neck)_(optimizer(no.))_(loss(no.))_(step or iter)_(imgsize)_(aug(no.))",
            "entity": "hype-squad",
        },
        interval=50,
        log_checkpoint=False,
        log_checkpoint_metadata=False,
        num_eval_images=100,
    ),
]

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# dataset 확인
print(datasets[0])

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

# inference
inference.run(checkpoint_dir + "/exp.py")
