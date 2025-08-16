import torch
import os

def ExportToOnnx(model, checkpoint, save_dir, input_size, device):
    """
    @brief 转换模型到onnx格式
    """
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
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
    # model = PPLiteSeg(num_class=2, n_channel=1, encoder_channels=[8, 16, 32, 64, 128],
    #                     encoder_type='stdc3', fusion_type='both', is_eval=False)
    # model = SegmentModel(1, 3, 4, 1, is_eval=True)
    # model = UNet(n_channels=1, n_classes=1, bilinear=True, is_eval=True)
    # model = CenterNet(in_channel=1, num_classes=1, is_eval=True)
    # checkpoint = r'Y:\SegmentModel\PointDetection\CheckPoint_20240627-011036\best_checkpoint_499_loss_1.02.pth'
    # model = LiteHRNet(num_class=4, n_channel=3, base_ch=16)
    # checkpoint = r'Y:\SegmentModel\PointDetection\CheckPoint\lite_hrnet.pth'
    # ExportToOnnx(model, checkpoint, 'Onnx', (1, 3, 256, 256), 'cpu')
    import cv2
    import numpy as np
    model_file = "export/lite_hrnet.onnx"
    model = cv2.dnn.readNetFromONNX(model_file)
    input = np.zeros(shape=(1, 3, 256, 256), dtype=np.float32)
    model.setInput(input)
    start = cv2.getTickCount()
    output = model.forward()
    print("inference loss: %d ms" % ((cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000))
    print(output.shape)
