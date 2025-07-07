"""
@brief 图像增强
"""
import albumentations as alb

class DataTransform(object):
    """数据增强"""
    def __init__(self, p: float = 0.5, input_size: int = 224):
        self.pixel_level = []
        self.spatial_level = []
        self.p = p
        self.input_size = input_size
        self.transform = alb.Compose(self.PixelLevelTransforms() +
                                     self.SpatialLevelTransforms())

    def PixelLevelTransforms(self):
        """像素级增强"""
        pixel_level = [alb.Defocus(radius=(1, 3), p=self.p),
                        alb.Downscale(scale_min=0.5, scale_max=0.85, p=self.p, interpolation=1),
                        alb.Emboss(p=self.p),
                        alb.GaussNoise(p=self.p),
                        alb.ZoomBlur(p=self.p, max_factor=(1.0, 1.1), step_factor=(0.1, 0.5)),
                        alb.MultiplicativeNoise(p=self.p),
                        alb.RandomBrightnessContrast(p=self.p),
                        alb.RandomGamma(p=self.p),
                        alb.UnsharpMask(p=self.p),
                        alb.HueSaturationValue(p=self.p)
                        ]
        return pixel_level

    def SpatialLevelTransforms(self):
        """空间级增强"""
        spatial_level = [alb.ElasticTransform(p=self.p),
                          alb.Flip(p=self.p),
                          alb.OpticalDistortion(p=self.p),
                          alb.ShiftScaleRotate(p=self.p, scale_limit=0.1, rotate_limit=15),
                          alb.RandomResizedCrop(height=self.input_size, width=self.input_size, p=self.p, scale=(0.8, 1.2)),
                          ]
        return spatial_level

    def __call__(self, image):
        """执行变换"""
        transformed = self.transform(image=image)
        return transformed['image']