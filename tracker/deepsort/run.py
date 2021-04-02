import os
import cv2
import time
import pathlib
import argparse
import torch
import warnings

import numpy as np

from tracker.deepsort import build_detector
from tracker.deepsort.parent.deep_sort import build_tracker
from tracker.deepsort.parent.utils.draw import draw_boxes
from tracker.deepsort.parent.utils.parser import get_config
from tracker.deepsort.parent.utils.log import get_logger
from tracker.deepsort.parent.utils.io import write_results


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = pathlib.Path(video_path)
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.classes =  self.active_cls_id = args.classes
        if args.classes is not None:
            self.active_cls_id = [i for i, item in enumerate(self.class_names) if item in args.classes]

    def __enter__(self):
        if self.args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            save_path = pathlib.Path(self.args.save_path)
            self.save_video_path = save_path / (self.video_path.stem + '.avi')
            self.save_results_path = save_path / (self.video_path.stem + '.txt')

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(str(self.save_video_path), fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(str(self.args.save_path)))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            if self.classes is not None:
                mask = np.isin(cls_ids, self.active_cls_id)
            else:
                mask = np.ones_like(cls_ids, dtype=np.bool)

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="tests/deepsort/MOT16-example.webm")
    parser.add_argument("--config_detection", type=str, default="tracker/deepsort/config/scaled_yolov4.yaml")
    parser.add_argument("--config_deepsort", type=str, default="tracker/deepsort/config/deep_sort.yaml")
    parser.add_argument('--classes', nargs='+', type=str, default=None, help='detect and track specific classes')
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="logs/deepsort_test")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.__dict__.pop('config_detection'))
    cfg.merge_from_file(args.__dict__.pop('config_deepsort'))

    with VideoTracker(cfg, args, video_path=args.video_path) as vdo_trk:
        vdo_trk.run()