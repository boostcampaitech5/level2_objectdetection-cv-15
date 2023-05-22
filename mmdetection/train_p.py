import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device
import inference
import wandb
import argparse


def train(args):
    wandb.login()

    base_dir = os.getcwd() + "/custom_configs"
    checkpoint_dir = os.getcwd() + f"/checkpoint/{args.model_name}"
    root = "../../dataset/"

    # load config file
    print(base_dir + args.model_config)
    cfg = Config.fromfile(base_dir + args.model_config)

    # model config 수정
    cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.model.roi_head.bbox_head.loss_bbox = dict(type="CIoULoss", loss_weight=12.0)

    # dataset config 수정
    # cfg.data.train.pipeline[14]["img_scale"] = args.resolution

    # cfg.data.val.ann_file = root + "train.json"
    # cfg.data.val.pipeline[1]["img_scale"] = args.resolution

    # cfg.data.test.pipeline[1]["img_scale"] = args.resolution
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
                "project": args.project,
                "name": args.model_name,
                "entity": "hype-squad",
                # "config": cfg,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="faster_rcnn_r50_fpn",
        help="model name, example) RESNET50_SGD_STEP_E12",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="/custom_1x.py",
        help="model config file",
    )
    parser.add_argument(
        "--resolution",
        type=tuple,
        default=(512, 512),
        help="resolution",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Faster_RCNN",
        help="project name",
    )
    args = parser.parse_args()
    print(args)

    train(args)
