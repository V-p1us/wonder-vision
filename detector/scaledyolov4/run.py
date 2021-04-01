import torch
import argparse

import detector.scaledyolov4.parent.detect as detect


DEFAULT_DIRS = {
    'weights': 'detector/scaledyolov4/weights/yolov4-csp.weights',
    'source': 'tests/scaled_yolov4',
    'output': 'logs/scaled_yolov4_test',
    'cfg': 'detector/scaledyolov4/parent/models/yolov4-csp.cfg',
    'names': 'detector/scaledyolov4/parent/data/coco.names'
}

def custom_detect(opt):
    # set arguments and run detect
    detect.opt = opt
    detect.detect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=[DEFAULT_DIRS['weights']], help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=DEFAULT_DIRS['source'], help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=DEFAULT_DIRS['output'], help='output folder')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default=DEFAULT_DIRS['cfg'], help='*.cfg path')
    parser.add_argument('--names', type=str, default=DEFAULT_DIRS['names'], help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        custom_detect(opt)
