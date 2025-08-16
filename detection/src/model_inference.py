import cv2
import numpy as np

def local_maximum_value(heat: np.ndarray, ksize: int = 3):
    """局部极值计算"""
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(ksize, ksize))
    keep = np.zeros_like(heat)
    b, c, h, w = heat.shape
    for i in range(b):
        for j in range(c):
            lmax_ij = cv2.morphologyEx(heat[i, j], cv2.MORPH_DILATE, kernel)
            keep_ij = (lmax_ij == heat[i, j]).astype(np.float32)
            keep[i, j] = keep_ij
    return heat * keep

def decode_bbox(heatmaps: np.ndarray, reg_whs: np.ndarray, reg_offsets: np.ndarray, confidence: float = 0.5):
    """
    @brief 模型预测bbox解码
    @param heatmaps     中心热图 shape = (b, c, h, w)
    @param reg_whs      高宽回归 shape = (b, 2, h, w)
    @param reg_offsets  中心偏差归回 shape = (b, 2, h, w)
    @return 检测框输出形式：x1, y1, x2, y2, score, class_id
    """
    # 提取热图局部极值
    heatmaps = local_maximum_value(heatmaps)
    # 记录检测结果
    detects = []
    batch, channel, output_h, output_w = heatmaps.shape
    for b in range(batch):
        pred_hmap = heatmaps[b].reshape([channel, -1]).swapaxes(0, 1)  # shape = (h * w, c)
        pred_wh = reg_whs[b].reshape([2, -1]).swapaxes(0, 1)  # shape = (h * w, 2)
        pred_offset = reg_offsets[b].reshape([2, -1]).swapaxes(0, 1)  # shape = (h * w, 2)
        # 生成坐标网格
        coord_xs, coord_ys = np.meshgrid(np.arange(0, output_h), np.arange(0, output_w))
        coord_xs = coord_xs.flatten().astype(np.float32)
        coord_ys = coord_ys.flatten().astype(np.float32)
        # 获取类别置信度
        class_conf = np.max(pred_hmap, axis=-1)
        # 获取类别id
        class_pred = np.argmax(pred_hmap, axis=-1).astype(np.float32)
        # 根据置信度过滤目标掩膜
        mask = class_conf > confidence
        if np.sum(mask) == 0:
            detects.append([])
            continue
        # 根据掩膜提取信息
        # 解码中心坐标
        pred_offset_mask = pred_offset[mask]
        coord_xs_mask = coord_xs[mask] + pred_offset_mask[..., 0]
        coord_ys_mask = coord_ys[mask] + pred_offset_mask[..., 1]
        coord_xs_mask = coord_xs_mask[..., np.newaxis]
        coord_ys_mask = coord_ys_mask[..., np.newaxis]
        # 解码高宽
        pred_wh_mask = pred_wh[mask] * 10
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        # 检测框输出形式：x1, y1, x2, y2, score, class_id
        bboxes = np.concatenate([coord_xs_mask - half_w,
                                 coord_ys_mask - half_h,
                                 coord_xs_mask + half_w,
                                 coord_ys_mask + half_h], axis=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0.0, 1.0)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0.0, 1.0)

        detect = np.concatenate([bboxes, class_conf[mask][..., np.newaxis], class_pred[mask][..., np.newaxis]],
                                axis=-1)
        detects.append(detect)

    return detects


class Detector:
    """检测器"""
    def __init__(self, onnx_file: str, in_channel: int = 3, input_size: tuple = (512, 512), num_classes: int = 1):
        self.model = cv2.dnn.readNetFromONNX(onnx_file)
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channel = in_channel

    def preprocess(self, input: np.ndarray):
        """前处理"""
        # 矩形填充
        scale = min(self.input_size[0] / input.shape[0], self.input_size[1] / input.shape[1])
        new_height = min(int(input.shape[0] * scale), self.input_size[0])
        new_width = min(int(input.shape[1] * scale), self.input_size[1])
        resize_input = cv2.resize(input, dsize=(new_width, new_height))
        if resize_input.ndim == 2:
            output = np.zeros(shape=self.input_size, dtype=np.uint8)
        else:
            output = np.zeros(shape=(self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        output[:new_height, :new_width] = resize_input.copy()
        return output

    def inference(self, input: np.ndarray):
        """推理"""
        if self.in_channel == 1:
            input_blob = input.copy()
            if input_blob.shape[0] != self.input_size[0] or input_blob.shape[1] != self.input_size[1]:
                input_blob = cv2.resize(input_blob, dsize=(self.input_size[1], self.input_size[0]))
            input_blob = input_blob[np.newaxis, np.newaxis, ...]
            input_blob = input_blob.astype(np.float32) / 255.0
        else:
            input_blob = cv2.dnn.blobFromImage(input, 1.0/255.0, (self.input_size[1], self.input_size[0]), swapRB=True)
        self.model.setInput(input_blob)
        output_blob = self.model.forward()
        return output_blob

    def postprocess(self, output_blob: np.ndarray, input_size: tuple, score_thresh: float = 0.5, nms_thresh: float = 0.5):
        """后处理"""
        heatmaps = output_blob[:, 0:self.num_classes, :, :]
        reg_whs = output_blob[:, self.num_classes: self.num_classes + 2, :, :]
        reg_offsets = output_blob[:, self.num_classes + 2: self.num_classes + 4, :, :]
        detections = decode_bbox(heatmaps, reg_whs, reg_offsets, score_thresh)
        if len(detections) == 0 or len(detections[0]) == 0:
            return None
        else:
            length = max(input_size)
            bboxes = detections[0][:, :4] * length
            bboxes = bboxes.astype(np.int32)
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, input_size[1] - 1)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, input_size[0] - 1)
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            scores = detections[0][:, 4]
            indices = detections[0][:, 5].astype(np.int32)
            nms_indices = cv2.dnn.NMSBoxesBatched(bboxes.tolist(), scores.tolist(), indices.tolist(),
                                                  score_thresh, nms_thresh)
            results = [dict(bbox=tuple(bboxes[i]), score=scores[i], index=indices[i]) for i in nms_indices]
            return results

    def __call__(self, image: np.ndarray, score_thresh: float = 0.5):
        """检测推理"""
        # 前处理
        input = self.preprocess(image)
        # 推理
        predict = self.inference(input)
        # 后处理
        outputs = self.postprocess(predict, image.shape[:2], score_thresh)
        return outputs

def generate_yolo_labels(image_dir: str, onnx_file: str):
    """生成yolo标注"""
    from tqdm import tqdm
    import os

    label_dir = os.path.join(os.path.dirname(image_dir), "labels")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print("create label_dir: ", label_dir)

    detector = Detector(onnx_file, in_channel=1)
    filenames = [name for name in os.listdir(image_dir) if name.split('.')[-1] in ('jpg', 'bmp', 'png', 'tif')]
    with tqdm(total=len(filenames), desc="process: ") as pbar:
        for file in filenames:
            image = cv2.imdecode(np.fromfile(os.path.join(image_dir, file), np.uint8), 0)
            results = detector(image)
            if results is None:
                continue
            height, width = image.shape[:2]
            labels = []
            for result in results:
                x, y, w, h = result['bbox']
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                id = result['index']
                label = "%d %.6f %.6f %.6f %.6f\n" % (id, cx, cy, nw, nh)
                labels.append(label)
            label_file = os.path.join(label_dir, file.replace(file.split(".")[-1], "txt"))
            with open(label_file, "w") as txt_file:
                txt_file.writelines(labels)

            pbar.set_postfix(filename=file)
            pbar.update()

if __name__ == "__main__":
    image_dir = r"..\data\droplet\images"
    onnx_file = r"export\centernet_128a.onnx"
    generate_yolo_labels(image_dir=image_dir, onnx_file=onnx_file)
    # import os
    # import matplotlib.pyplot as plt
    # model_path = r"export\centernet_128a.onnx"
    # detector = Detector(model_path, in_channel=1)
    # data_dir = r"D:\WorkSpace\my_project\segment_test\data\images"
    # filenames = [name for name in os.listdir(data_dir) if name.split('.')[-1] in ('jpg', 'bmp', 'png', 'tif')]
    # for file in filenames:
    #     image = cv2.imdecode(np.fromfile(os.path.join(data_dir, file), np.uint8), 0)
    #     results = detector(image)
    #     if results is None:
    #         continue
    #
    #     show_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     for result in results:
    #         x, y, w, h = result['bbox']
    #         cv2.circle(show_image, (x + w // 2, y + h // 2), int((w + h) / 4), (0, 255, 0), 1)
    #     plt.imshow(show_image)
    #     plt.show()