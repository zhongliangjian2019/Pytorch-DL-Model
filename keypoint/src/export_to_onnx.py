import torch
import os
from model import KeyPointModel

def ExportToOnnx(model, checkpoint, save_dir, input_size, device):
    """
    @brief 转换模型到onnx格式
    """
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    input = torch.randn(input_size, device=device)
    onnx_file = os.path.join(save_dir, os.path.basename(checkpoint).replace('pth', 'onnx'))

    torch.onnx.export(model,    # 模型
                        input,    # 输入
                        onnx_file, # 输出路径
                        input_names=["input"],
                        output_names=["output"],
                        verbose=True,
                        opset_version=11)

    return onnx_file

if __name__ == '__main__':
    model = KeyPointModel(in_channel=3, num_classes=1, is_eval=True)
    checkpoint = r'../checkpoints/qrcode_keypoint_best.pth'
    ExportToOnnx(model, checkpoint, '../export', (1, 3, 128, 128), 'cpu')
