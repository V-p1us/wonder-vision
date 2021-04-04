from copy import deepcopy

from detector.scaled_yolov4.parent.utils.general import scale_coords


def rescale_detections(pred, orig_shape, input_shape):
    """
    Rescale boxes from processed frame coord to original frame coord
    """
    processed_pred = deepcopy(pred)
    for i, det in enumerate(pred):
        if det is not None and len(det):
            processed_pred[i][:, :4] = scale_coords(input_shape[2:], det[:, :4], orig_shape).round()
    return processed_pred
    