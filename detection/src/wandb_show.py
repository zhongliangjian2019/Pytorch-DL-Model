import numpy as np
import wandb

def wandb_boxes2d(image: np.ndarray, p_bboxes, g_bboxes, class_labels: dict):
    """可视化boxes2d"""
    img = wandb.Image(
        np.transpose(image, axes=(1, 2, 0)),
        boxes={
            "predictions":{
                "box_data": [{"position": {"minX": float(box[0]), "maxX": float(box[2]), "minY": float(box[1]), "maxY": float(box[3])},
                              "class_id": int(box[5]),
                              "box_caption": class_labels[int(box[5])],
                              "score": float(box[4])} for box in p_bboxes],
                "class_labels": class_labels,
            },
            "ground_truth":{
                 "box_data": [{"position": {"minX": float(box[0]), "maxX": float(box[2]), "minY": float(box[1]), "maxY": float(box[3])},
                              "class_id": int(box[5]),
                              "box_caption": class_labels[int(box[5])],
                              "score": float(box[4])} for box in g_bboxes],
                "class_labels": class_labels,
            }
        },
    )
    return img
