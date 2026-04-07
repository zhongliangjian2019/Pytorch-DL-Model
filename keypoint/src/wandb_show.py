"""
wandb工具可视化
"""
import numpy as np
import wandb

def wandb_boxes2d(image: np.ndarray, p_bboxes, g_bboxes, class_labels: dict, radius: int = 11):
    """可视化boxes2d"""
    img = wandb.Image(
        image,
        boxes={
            "predictions":{
                "box_data": [{"position": {"minX": float(box[0] - radius / 2), "maxX": float(box[0] + radius / 2),
                                           "minY": float(box[1] - radius / 2), "maxY": float(box[1] + radius / 2)},
                              "class_id": int(box[3]),
                              "box_caption": class_labels[int(box[3])],
                              "score": float(box[2])} for box in p_bboxes],
                "class_labels": class_labels,
            },
            "ground_truth":{
                 "box_data": [{"position": {"minX": float(box[0] - radius / 2), "maxX": float(box[0] + radius / 2),
                                           "minY": float(box[1] - radius / 2), "maxY": float(box[1] + radius / 2)},
                              "class_id": int(box[3]),
                              "box_caption": class_labels[int(box[3])],
                              "score": float(box[2])} for box in g_bboxes],
                "class_labels": class_labels,
            }
        },
    )
    return img
