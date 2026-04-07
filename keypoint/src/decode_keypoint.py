"""
关键点位置解码
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_keypoints(pred_hms, pred_offsets, confidence, cuda):
    """解码关键点"""
    pred_hms = pool_nms(pred_hms)
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    # -------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    # -------------------------------------------------------------------------#
    for batch in range(b):
        # -------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        # -------------------------------------------------------------------------#
        heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        # -------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        # -------------------------------------------------------------------------#
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        # -------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        # -------------------------------------------------------------------------#
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > confidence

        # -----------------------------------------#
        #   取出得分筛选后对应的结果
        # -----------------------------------------#
        pred_offset_mask = pred_offset[mask]
        #   计算调整后预测框的中心
        # ----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        # ----------------------------------------#
        #   获得预测框的左上角和右下角
        #   检测框输出形式：x, y, score, class_id
        # ----------------------------------------#
        bboxes = torch.cat([xv_mask, yv_mask], dim=1)
        bboxes[:, 0] /= output_w
        bboxes[:, 1] /= output_h
        detect = torch.cat(
            [bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
        detects.append(detect)

    return detects

def decode_bbox_cpu(pred_hms: np.ndarray, pred_offsets: np.ndarray, confidence: float = 0.5):
    # -------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128,
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    # -------------------------------------------------------------------------#
    def pool_nms_cpu(heat: np.ndarray, ksize=3):
        """局部极值计算：opencv版本"""
        import cv2
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(ksize, ksize))
        keep = np.zeros_like(heat)
        b, c, h, w = heat.shape
        for i in range(b):
            for j in range(c):
                hmax_ij = cv2.morphologyEx(heat[i, j, :, :], cv2.MORPH_DILATE, kernel)
                keep_ij = (hmax_ij == heat[i, j, :, :]).astype(np.float32)
                keep[i, j, :, :] = keep_ij
        return heat * keep

    # 提取热图局部极值
    pred_hms = pool_nms_cpu(pred_hms)

    # 记录检测结果
    detects = []
    b, c, output_h, output_w = pred_hms.shape
    for batch in range(b):
        heat_map = pred_hms[batch].reshape([c, -1]).swapaxes(0, 1)
        pred_offset = pred_offsets[batch].reshape([2, -1]).swapaxes(0, 1)
        xv, yv = np.meshgrid(np.arange(0, output_h), np.arange(0, output_w))
        xv, yv = xv.flatten(), yv.flatten()
        xv = xv.astype(np.float32)
        yv = yv.astype(np.float32)
        class_conf = np.max(heat_map, axis=-1)
        if c == 1:
            class_pred = np.zeros_like(heat_map, dtype=np.float32).flatten()
        else:
            class_pred = np.argmax(heat_map, axis=-1).astype(np.float32)

        # 获取置信度大于阈值的目标
        mask = class_conf > confidence
        pred_offset_mask = pred_offset[mask]
        xv_mask = xv[mask] + pred_offset_mask[..., 0]
        yv_mask = yv[mask] + pred_offset_mask[..., 1]
        xv_mask = xv_mask[..., np.newaxis]
        yv_mask = yv_mask[..., np.newaxis]

        # 检测框输出形式：x, y, score, class_id
        points = np.concatenate([xv_mask, yv_mask], axis=1)
        points[:, 0] /= output_w
        points[:, 1] /= output_h
        detect = np.concatenate(
            [points, class_conf[mask][..., np.newaxis], class_pred[mask][..., np.newaxis]], axis=-1)
        detects.append(detect)

    return detects

def draw_cross(img, center: tuple, length: int, color: tuple, width: int = 1):
    """绘制十字标记"""
    cv2.line(img, (center[0], center[1] - length), (center[0], center[1] + length), color, width)
    cv2.line(img, (center[0] - length, center[1]), (center[0] + length, center[1]), color, width)
    return img

if __name__ == "__main__":
    import BaiseToolFunc as BTF
    import matplotlib.pyplot as plt
    import os

    model_path = r".\Onnx\centernet_128a.onnx"
    model = cv2.dnn.readNetFromONNX(model_path)

    # model = CenterNet(in_channel=1, num_classes=1)
    # weights = torch.load("./CheckPoint/point_detection_50.pth", map_location="cpu")
    # model.load_state_dict(weights)
    # model.eval()

    data_dir = r"Y:\SegmentModel\dataset\droplet\20240514"
    test_file = os.path.join(data_dir, "test.txt")
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            filenames = [line.strip().split('.')[0] for line in f.readlines() if len(line) != 0]
    for file in filenames:
        image_path = os.path.join(data_dir, "images", file + ".jpg")
        image = BTF.ReadImage(image_path, 0)

        # with torch.no_grad():
            # input_data = image[np.newaxis, np.newaxis, ...] / 255.0
            # input_data = torch.tensor(input_data, dtype=torch.float32)
            # output_data = model(input_data)
            # pred_hms, pred_whs, pred_offsets = output_data
            # pred_hms = pred_hms.cpu().numpy()
            # pred_whs = pred_whs.cpu().numpy()
            # pred_offsets = pred_offsets.cpu().numpy()

        input_blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (512, 512))
        model.setInput(input_blob)
        output_blob = model.forward()
        print(output_blob.shape)
        pred_hms = output_blob[:, 0, :, :]
        pred_hms = pred_hms[:, np.newaxis, :, :]
        print(pred_hms.shape)
        pred_whs = output_blob[:, 1:3, :, :]
        print(pred_whs.shape)
        pred_offsets = output_blob[:, 3:, :, :]
        print(pred_offsets.shape)

        # plt.subplot(1, 2, 1)
        # plt.imshow(pred_offsets[0, 0, :, :])
        # plt.subplot(1, 2, 2)
        # plt.imshow(pred_offsets[0, 1, :, :])
        # plt.show()

        detects = decode_bbox_cpu(pred_hms, pred_whs, pred_offsets, 0.25)
        show_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for detect in detects:
            if len(detect) != 0:
                detect[:, [0, 2]] *= image.shape[1]
                detect[:, [1, 3]] *= image.shape[0]
                for i in range(detect.shape[0]):
                    box = detect[i, :4].astype(np.int32)
                    score = detect[i, 4]
                    # cla_id = detect[i, 5]
                    # cv2.rectangle(show_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                    center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                    radius = int(((box[2] - box[0]) + (box[3] - box[1])) / 4)
                    cv2.circle(show_image, center, radius, (0, 200, 0), 1)
                    draw_cross(show_image, center, 2, (0, 200, 0), 1)
                    # text = "%d" % radius
                    # font_face = cv2.FONT_HERSHEY_TRIPLEX
                    # font_scale = 0.3
                    # thickness = 1
                    # size = cv2.getTextSize(text, fontFace=font_face, fontScale=font_scale, thickness=thickness)
                    # cv2.putText(show_image, text, (center[0] - size[0][0] // 2, center[1] + size[0][1] // 2), font_face,
                    #             font_scale, (200, 0, 0), thickness)
        # pred_offsets[:, :, :, :] = 0
        # detects = decode_bbox_cpu(pred_hms, pred_whs, pred_offsets, 0.25)
        # for detect in detects:
        #     if len(detect) != 0:
        #         detect[:, [0, 2]] *= image.shape[1]
        #         detect[:, [1, 3]] *= image.shape[0]
        #         for i in range(detect.shape[0]):
        #             box = detect[i, :4].astype(np.int32)
        #             score = detect[i, 4]
        #             # cla_id = detect[i, 5]
        #             # cv2.rectangle(show_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        #             center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        #             radius = int(((box[2] - box[0]) + (box[3] - box[1])) / 4)
        #             cv2.circle(show_image, center, radius, (200, 0, 0), 1)
        #             draw_cross(show_image, center, 2, (200, 0, 0), 1)

        plt.imshow(show_image)
        plt.show()


