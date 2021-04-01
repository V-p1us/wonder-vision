# include submodule paths
import cv2
import torch
import logging

import numpy as np

from detector.scaledyolov4.parent.utils.datasets import letterbox
from detector.scaledyolov4.parent.detect import load_classes
from detector.scaledyolov4.parent.models.models import Darknet, load_darknet_weights
from detector.scaledyolov4.parent.utils.general import non_max_suppression, xyxy2xywh
from detector.scaledyolov4.tools import rescale_detections


class ScaledYOLOv4(object):
    def __init__(self, cfgfile, weightfile, namesfile, img_size=640,
                 conf_thres=0.4, iou_thres=0.5, is_xywh=False,
                 use_cuda=None, half=False):
        # net definition
        self.net = Darknet(cfgfile, img_size)
        load_darknet_weights(self.net, weightfile)
        logger = logging.getLogger("root.tracker.deepsort.scaledyolov4")
        logger.info('Loading weights from %s... Done!' % (weightfile))
        if use_cuda is None:
            use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda'
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.size = (img_size,) * 2 if isinstance(img_size, int) else img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        self.num_classes = self.net.module_defs[-1]['classes']
        self.class_names = load_classes(namesfile)
        self.dtype = torch.float16 if half else torch.float32

    def __call__(self, ori_img):
        """
        arguments:
            ori_img - single RGB image (HxWxC)
        """
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        img, *_ = letterbox(ori_img, self.size)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float) / 255.

        # img = cv2.resize(img, self.size).transpose(2, 0, 1)

         # forward
        with torch.no_grad():
            img = torch.tensor(img, device=self.device, dtype=self.dtype).unsqueeze(0)
            # Inference
            pred = self.net(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       classes=None, agnostic=False)
            # Process detections
            boxes = rescale_detections(pred, ori_img.shape, img.shape)[0].cpu()

        if len(boxes) == 0:
            bbox = torch.FloatTensor([]).reshape([0, 4])
            cls_conf = torch.FloatTensor([])
            cls_ids = torch.LongTensor([])
        else:
            height, width = ori_img.shape[:2]
            bbox = boxes[:, :4]
            if self.is_xywh:
                bbox = xyxy2xywh(bbox)

            # bbox *= torch.FloatTensor([[width, height, width, height]])
            cls_conf = boxes[:, -2]
            cls_ids = boxes[:, -1].long()
        return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()


def demo():
    import os
    from vizer.draw import draw_boxes

    yolo = ScaledYOLOv4('detector/scaledyolov4/parent/models/yolov4-csp.cfg',
                        'detector/scaledyolov4/weights/yolov4-csp.weights',
                        'detector/scaledyolov4/parent/data/coco.names')
    print("yolo.size =", yolo.size)
    root = "detector/scaledyolov4/test"
    resdir = os.path.join(root, "results")
    os.makedirs(resdir, exist_ok=True)
    files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.jpg')]
    files.sort()
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, cls_conf, cls_ids = yolo(img)

        if bbox is not None:
            img = draw_boxes(img, bbox, cls_ids, cls_conf, class_name_map=yolo.class_names)
        # save results
        cv2.imwrite(os.path.join(resdir, os.path.basename(filename)), img[:, :, (2, 1, 0)])
        # display results
        # cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("yolo", 600,600)
        # cv2.imshow("yolo",res[:,:,(2,1,0)])
        # cv2.waitKey(0)


if __name__ == "__main__":
    demo()
