import os
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from utils import coco_to_pascal, make_submission


def run(config_path):
    # config 파일 load
    cfg = Config.fromfile(config_path)

    # dataset loading
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset=dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # model, checkpoint build
    model_path = os.path.join(cfg.work_dir, "latest.pth")

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, model_path, map_location="cpu")

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # inference
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    out_prediction_strings, out_file_names = coco_to_pascal(cfg, output)
    make_submission(cfg, out_prediction_strings, out_file_names)


if __name__ == "__main__":
    run("/opt/ml/WorkSpace/checkpoint/faster_rcnn_r50_fpn_1x_coco_trash/exp.py")
