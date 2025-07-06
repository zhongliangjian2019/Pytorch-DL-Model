import cv2
import torch
from model_zoo import UNet
import tool_func as tf

class Segmentor(object):
    """分割模型推理"""
    def __init__(self, model_path: str, input_size: tuple):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.input_size = input_size

    def load_model(self, model_path: str):
        """加载模型"""
        model = UNet(n_channels=3, n_classes=1, bilinear=True, is_eval=True)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        model.to(device=self.device)
        return model

    @tf.time_metric
    def inference(self, images: list):
        """推理"""
        input_blob = cv2.dnn.blobFromImages(images, 1.0/255.0,
                                            size=(self.input_size[1], self.input_size[0]), swapRB=True)
        with torch.no_grad():
            input_blob = torch.as_tensor(input_blob, dtype=torch.float32, device=self.device)
            output_blob = self.model(input_blob)

        output_blob = output_blob.detach().cpu().numpy()
        return output_blob

if __name__ == "__main__":
    """模块测试"""
    image_path = "../data/images"
    model_path = "../checkpoints/best.pth"
    image = tf.ReadImage(image_path, 1)
    model = Segmentor(model_path, input_size=(640, 640))
    for i in range(100):
        output_blob = model.inference([image, image])
    pass