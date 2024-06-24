import torch
from IoU import intersection_over_union

"""
    1. Select next highest-scoring box
    2. Eliminate lower-scoring boxes with IoU > threshold (e.g. 0.7)
    3. If any boxes remain, GOTO 1
    
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        prob_threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
 """


def nms(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    assert type(bboxes) == list

    # bboxes = [[1, 0.9, x1, y1, x2, y2]]

    # let's say that we have 5 bounding boxes for our object and here is the probability score that our object is
    # within this bbox: p1 = 0.6, p2 = 0.9, p3 = 0.35, p4 = 0.8, p5 = 0.2 and our threshold is set to 0.5
    # so in our case we take only the bbox that have probability scores higher than 0.5
    # in our case, we take bbox1, bbox2, and bbox4 and (re-)save them in our bboxes
    # [0.6, 0.9, 0.8] for probability scores which are in the first index

    bboxes = [box for box in bboxes if box[1] > prob_threshold]  # box[1] is probability score

    # then we sort by these probability scores and get
    # [0.9, 0.8, 0.6] because we wanna first choose the box with higher probability

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # sort by probability score

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or
                  intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format)
                  < iou_threshold]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
