"""
功能描述：模型导出
"""
import torch
import os
from model_zoo import MobileNetV3_Small

def ExportToOnnx(model, checkpoint, save_dir, input_size, device):
    """
    功能描述：转换模型到onnx格式
    :param model:
    :param checkpoint:
    :param save_dir:
    :param input_size:
    :return: onnx_file
    """
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input = torch.randn(input_size, device=device)
    onnx_file = os.path.join(save_dir, os.path.basename(checkpoint).replace('pth', 'onnx'))
    torch.onnx.export(model, input, onnx_file,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=11)

    return onnx_file

if __name__ == "__main__":
    """模块测试"""
    checkpoint = "../ckpts/mbv3_small.pth"
    model = MobileNetV3_Small(num_classes=1000, in_channel=3, is_eval=True)
    ExportToOnnx(model, checkpoint, '../export', (1, 3, 224, 224), device='cpu')
