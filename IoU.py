"""
Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
"""
import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    # boxes_pred shape is (N, 4) and boxes_labels shape is (N, 4)
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Getting corner point
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)

    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # intersection:
    # .clamp(0) is for edge case when the boxes don't intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


"""
Object localization is finding what and where a (single) object exists in an image.

Object detection is finding what and where (multiple) objects are in an image.

a common way to define bounding boxes is to use x1, y1, x2, x2 or x1, y1, h, w

(x1, y1) is the upper left corner point, (x2, y2) is the bottom right corner point

Image --> CNN --> class prediction, (x1, y1, x2, y2)

IoU (Intersection over Union)

Area of Intersection / Area of Union = [0, 1] range

IoU > 0.5 "decent"
IoU > 0.7 "pretty good"
IoU > 0.9 "almost perfect"

Box1 = [x1, y1, x2, y2]
Box2 = [x1, y1, x2, y2]

upper left is the origin (0, 0)

x1 is max(box1[0], box2[0])
y1 is max(box1[1], box2[1])

x2 is min(box1[2], box2[2])
y2 is min(box1[3], box2[3])
"""