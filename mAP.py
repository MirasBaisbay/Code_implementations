import torch
from collections import Counter

from IoU import intersection_over_union


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # List to store Average Precision (AP) for each class.
    average_precisions = []

    # Small value to avoid division by zero.
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Separate detections and ground_truths belonging to class c.
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # Use Counter to count the number of ground truth boxes per image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Convert counts to tensors for easier manipulation
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort detections by their confidence score in descending order
        detections.sort(key=lambda x: x[2], reverse=True)
        # Initialize TP and FP arrays to zeros
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        # Loop through each detection and calculate IoU with ground truth boxes of the same image
        for detection_idx, detection in enumerate(detections):

            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Determine if the detection is a TP or FP based on IoU and previously matched ground truths
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1
        # Compute cumulative sums of TP and FP.
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        # Calculate recall and precision at each detection.
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        # Concatenate extra points for numerical stability.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # Use the trapezoidal rule to integrate the area under the precision-recall curve (AUC).
        average_precisions.append(torch.trapz(precisions, recalls))
    # Average the APs for all classes to get the mean average precision.
    return sum(average_precisions) / len(average_precisions)
