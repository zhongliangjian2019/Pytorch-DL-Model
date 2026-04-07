"""
@brief 数据增强策略
@date 2023/07/14
@edit zlj-6883
"""

import albumentations as alb

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
                                     keypoint_params=alb.KeypointParams(format='xy', label_fields=['class_labels']))

    def PixelLevelTransforms(self):
        self.pixel_level = [
                            alb.Defocus(radius=(1, 3), p=self.p),
                            alb.Downscale(p=self.p),
                            alb.Emboss(p=self.p),
                            # alb.GaussNoise(p=self.p),
                            alb.MultiplicativeNoise(p=self.p),
                            alb.RandomBrightnessContrast(p=self.p),
                            alb.RandomGamma(p=self.p),
                            alb.UnsharpMask(p=self.p),
                            alb.HueSaturationValue(p=self.p)
                            ]

    def SpatialLevelTransforms(self):
        self.spatial_level = [
                              alb.VerticalFlip(self.p),
                              alb.HorizontalFlip(self.p),
                              alb.Affine(p=self.p, scale=(0.75, 1.0), rotate=(-15, 15))
                              ]

    def __call__(self, image, keypoints: list, class_labels: list):
        transformed = self.transform(image=image, keypoints=keypoints, class_labels=class_labels)
        return transformed['image'], transformed['keypoints'], transformed['class_labels']