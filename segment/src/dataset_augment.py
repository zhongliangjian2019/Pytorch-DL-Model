"""
@brief 分割模型训练数据增强
"""

import albumentations as alb
import numpy as np

class DataTransform(object):
    """数据增强"""
    def __init__(self, p: float = 0.5, width: int = 512, height: int = 512):
        self.pixel_level = []
        self.spatial_level = []
        self.p = p
        self.height = height
        self.width = width
        self.PixelLevelTransforms()
        self.SpatialLevelTransforms()
        self.transform = alb.Compose(self.pixel_level + self.spatial_level)

    def PixelLevelTransforms(self):
        """像素变换增强"""
        self.pixel_level = [alb.Defocus(radius=(1, 3), p=self.p),
                            alb.Downscale(scale_min=0.5, scale_max=0.85, p=self.p, interpolation=1),
                            alb.Emboss(p=self.p),
                            alb.GaussNoise(p=self.p),
                            alb.ZoomBlur(p=self.p, max_factor=(1.0, 1.1), step_factor=(0.1, 0.5)),
                            alb.MultiplicativeNoise(p=self.p),
                            alb.RandomBrightnessContrast(p=self.p),
                            alb.RandomGamma(p=self.p),
                            alb.UnsharpMask(p=self.p),
                            ]

    def SpatialLevelTransforms(self):
        """空间变换增强"""
        self.spatial_level = [
            alb.ElasticTransform(p=self.p),
            alb.VerticalFlip(self.p),
            alb.HorizontalFlip(self.p),
            alb.OpticalDistortion(p=self.p),
            alb.Affine(p=self.p, rotate=(-15, 15))]

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """执行增强处理"""
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']