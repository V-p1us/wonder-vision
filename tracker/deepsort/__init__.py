from tracker.deepsort.scaled_yolov4 import ScaledYOLOv4

__all__ = ['build_detector']


def build_detector(cfg, use_cuda):
    if cfg.ID == 'ScaledYOLOv4':
        return ScaledYOLOv4(cfgfile=cfg.CONFIG,
                            weightfile=cfg.WEIGHTS,
                            namesfile=cfg.CLASS_NAMES,
                            img_size=cfg.IMG_SIZE,
                            conf_thres=cfg.CONF_THRES,
                            iou_thres=cfg.IOU_THRES,
                            is_xywh=True,
                            use_cuda=use_cuda,
                            half=False)
    else:
        raise NotImplementedError
