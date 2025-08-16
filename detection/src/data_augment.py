"""
@brief 数据增强
@date 2023/07/14
@edit zlj-6883
"""
import albumentations as alb
import numpy as np

class DataTransform(object):
    def __init__(self, p: float = 0.5, width: int = 1824, height: int = 1824):
        self.pixel_level = []
        self.spatial_level = []
        self.p = p
        self.height = height
        self.width = width
        self.PixelLevelTransforms()
        self.SpatialLevelTransforms()
        self.transform = alb.Compose(self.pixel_level + self.spatial_level,
                                     bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels'],
                                                                min_visibility=0.5, clip=True, check_each_transform=False))

    def PixelLevelTransforms(self):
        self.pixel_level = [alb.Defocus(radius=(1, 3), p=self.p),
                            alb.ZoomBlur(p=self.p, max_factor=(1.0, 1.1), step_factor=(0.1, 0.5)),
                            alb.MultiplicativeNoise(p=self.p),
                            alb.RandomBrightnessContrast(p=self.p),
                            alb.RandomGamma(p=self.p),
                            ]

    def SpatialLevelTransforms(self):
        self.spatial_level = [
                              alb.HorizontalFlip(p=self.p),
                              alb.VerticalFlip(p=self.p),
                              alb.ShiftScaleRotate(p=self.p, scale_limit=(0.0, 1.0), rotate_limit=15),
                              alb.RandomResizedCrop(height=self.height, width=self.width, p=self.p, scale=(0.75, 1.0)),
                              ]

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, class_labels: np.ndarray):
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed['image'], transformed['bboxes'], transformed['class_labels']