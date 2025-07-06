"""
功能描述：DNN基础函数
"""
import cv2
import os
import numpy as np
import base64
from PIL import Image
import io
import math
import shutil

DOUBLEEPSILON = 1e-6  # 浮点数最小间隙

class Point2d(object):
    """二维整数点"""
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __add__(self, other):
        """加法"""
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """减法"""
        return Point2d(self.x - other.x, self.y - other.y)

    def __repr__(self):
        """打印"""
        return "Point2d(%d, %d)" % (self.x, self.y)

    def distance(self, other):
        """计算两点间的欧式距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class Point2f(object):
    """二维实数点"""
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        """加法"""
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """减法"""
        return Point2d(self.x - other.x, self.y - other.y)

    def distance(self, other):
        """计算两点间的欧式距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

def LoadModel(model_file: str):
    """
    @brief 加载网络模型
    @param model_file: 模型文件
    @return: model：加载的模型
    """
    assert os.path.exists(model_file), "ModelFile file not find"

    if model_file.split('.')[-1] == 'onnx':
        model = cv2.dnn.readNetFromONNX(model_file)
    else:
        model = cv2.dnn.readNetFromTensorflow(model_file)
    return model

def Softmax(input):
    """
    @brief softmax函数
    @param input: shape is [Batch, Class]
    @return: output: shape is [Batch, Class]
    """
    # 初始化输出
    output = np.zeros_like(input)
    # 获取行最大值
    input_max = [0.0 for i in range(input.shape[0])]
    for row in range(input.shape[0]):
        for col in range(input.shape[1]):
            if input[row, col] > input_max[row]:
                input_max[row] = input[row, col]

    # 按行求和
    sum = [0.0 for i in range(input.shape[0])]
    for row in range(input.shape[0]):
        for col in range(input.shape[1]):
            output[row, col] = math.exp(input[row, col] - input_max[row])
            sum[row] += output[row, col]

    # 归一化
    for row in range(input.shape[0]):
        for col in range(input.shape[1]):
            output[row, col] /= sum[row]

    return output

def Argmax(input):
    """
    @brief argmax函数
    @param input: shape is [Batch, Class]
    @param output: shape is [Batch, 1]
    @return: None
    """
    # 初始化输出
    output = []

    for row in range(input.shape[0]):
        max_value = -1
        max_index = -1
        for col in range(input.shape[1]):
            if input[row, col] > max_value:
                max_value = input[row, col]
                max_index = col
        output.append(max_index)

    return output

def GetContourCenters(contours):
    """获取轮廓质心（多个）
    :param contours 轮廓数组[n, 2]
    :return centers 质心数组 List[Point2d]
    """
    centers = []

    for i in range(len(contours)):
        center = GetContourCenter(contours[i])
        centers.append(center)

    return centers

def GetContourCenter(contour):
    """获取轮廓质心（单个）"""
    contour = np.reshape(contour, (-1, 2))
    # 计算轮廓矩
    mu = cv2.moments(contour)

    # 计算质心（中心）
    if (-DOUBLEEPSILON <= mu['m00'] <= DOUBLEEPSILON):
        # 取平均值
        cx = 0
        cy = 0
        for i in range(len(contour)):
            cx += contour[i, 0]
            cy += contour[i, 1]
        cx /= len(contour)
        cy /= len(contour)
    else:
        cx = mu['m10'] / mu['m00']
        cy = mu['m01'] / mu['m00']

    mc = Point2d(int(cx), int(cy))

    return mc

def MoveImage(image, step: tuple = (0, 0)):
    """
    @brief 平移图像
    @param image: 输入图像
    @param step: x和y方向的平移值平移值
    @return dst_image: 平移后的图像
    """
    dst_image = np.zeros_like(image, np.uint8)
    step_x, step_y = step[0], step[1]
    if step_x > 0:
        dst_image[:, step_x:] = image[:, :-step_x].copy()
    elif step_x < 0:
        dst_image[:, :step_x] = image[:, -step_x:].copy()
    else:
        dst_image = image.copy()

    image = dst_image.copy()
    dst_image[:, :] = 0

    if step_y > 0:
        dst_image[step_y:, :] = image[:-step_y, :].copy()
    elif step_y < 0:
        dst_image[:step_y, :] = image[-step_y:, :].copy()
    else:
        dst_image = image.copy()

    return dst_image

def ShowImage(image, winscale: float = 1.0, winname: str = 'test', wait: int = 0):
    """显示单张图像"""
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    winsize = (int(image.shape[0] * winscale), int(image.shape[1] * winscale))
    cv2.resizeWindow(winname, winsize[1], winsize[0])
    cv2.imshow(winname, image)

    if cv2.waitKey(wait) == 27:
        cv2.destroyWindow(winname)
        exit(0)
    else:
        cv2.destroyWindow(winname)

def ShowImages(image_list: object, winscale: float = 1.0, winname: str = 'test', wait: int = 0) -> object:
    """显示多张图像"""
    if not isinstance(image_list, list):
        show_image = image_list
        ShowImage(show_image, winscale, winname, wait)
    else:
        show_list = []
        # 统一图像为BGR
        for i, image in enumerate(image_list):
            if image.ndim == 2:
                show_list.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            else:
                show_list.append(image)

        show_image = show_list.pop(0)
        for image in show_list:
            show_image = np.hstack([show_image, image])
        ShowImage(show_image, winscale, winname, wait)

    return show_image

def CropImage(image, dsize: tuple = None):
    """裁剪图像为正方形"""
    if image is None:
        return None
    if dsize is None:
        if image.shape[0] == image.shape[1]:
            dst_image = image
        else:
            min_value = image.shape[1] if (image.shape[0] > image.shape[1]) else image.shape[0]
            crop_row = (image.shape[0] - min_value) // 2
            crop_col = (image.shape[1] - min_value) // 2
            dst_image = image[crop_row : image.shape[0] - crop_row, crop_col : image.shape[1] - crop_col].copy()
    else:
        dst_image = np.zeros(shape=dsize, dtype=np.uint8)
        if image.ndim == 3:
            dst_image = cv2.cvtColor(dst_image, cv2.COLOR_GRAY2BGR)

        crop_row = (image.shape[0] - dsize[0]) // 2
        crop_col = (image.shape[1] - dsize[1]) // 2
        if crop_row >= 0 and crop_col >= 0:
            dst_image = image[crop_row : dsize[0] + crop_row, crop_col : dsize[1] + crop_col].copy()
        elif crop_row < 0 and crop_col < 0:
            dst_image[-crop_row:crop_row, -crop_col:crop_col] = image.copy()
        elif crop_row < 0:
            dst_image[-crop_row:crop_row, :] = image[:, crop_col : dsize[1] + crop_col].copy()
        else:
            dst_image[:, -crop_col:crop_col] = image[crop_row : dsize[0] + crop_row, :].copy()
    return dst_image

def SplitBlockImage(input_image, size: tuple = (512, 152), pad_mode: str = 'edge'):
    """图像分块"""
    # 计算分块数量
    block_row = math.ceil(input_image.shape[0] / size[0])
    block_col = math.ceil(input_image.shape[1] / size[1])

    # 填充图像到指定大小
    image = np.zeros(shape=(size[0] * block_row, size[1] * block_col), dtype=np.uint8) + np.min(input_image)
    if pad_mode == 'edge':
        row = (image.shape[0] - input_image.shape[0]) // 2
        col = (image.shape[1] - input_image.shape[1]) // 2
        image[row : input_image.shape[0] + row, col : input_image.shape[1] + col] = input_image.copy()

    if pad_mode == 'left':
        row = (image.shape[0] - input_image.shape[0])
        col = (image.shape[1] - input_image.shape[1])
        image[row: input_image.shape[0] + row, col : input_image.shape[1] + col] = input_image.copy()

    if pad_mode == 'right':
        image[0: input_image.shape[0], 0: input_image.shape[1]] = input_image.copy()

    # 裁剪图像
    block_images = []
    for row in range(block_row):
        row_range = (row * size[0], (row + 1) * size[0])
        for col in range(block_col):
            col_range = (col * size[1], (col + 1) * size[1])
            block_image = image[row_range[0]:row_range[1], col_range[0]:col_range[1]].copy()
            block_images.append(block_image)

    return block_images, (block_row, block_col)

def CropImage5(in_image):
    """裁剪图像"""
    crop_images = []
    crop_images.append(CropImage(in_image, dsize=(25, 25)))
    block_images, _ = SplitBlockImage(in_image, size=(25, 25))
    crop_images.extend(block_images)
    return crop_images

def GetImageRoi(image: np.ndarray, center: Point2d, dsize: int):
    """根据中心及半径位置获取roi区域
    注意：返回值为原图像的引用，对返回值的直接操作将应用到原图
    """
    row = int(center.y)
    col = int(center.x)
    offset = dsize // 2
    row_range = slice(row - offset, row + offset + 1)
    col_range = slice(col - offset, col + offset + 1)

    if row_range.start < 0 or row_range.stop >= image.shape[0] or \
       col_range.start < 0 or col_range.stop >= image.shape[1]:
        # 超出索引边界
        return None
    else:
        return image[row_range, col_range]

def img_arr_to_b64(img_arr, image_format: str):
    '''转换np.array到json存储图像数据'''
    img_pil = Image.fromarray(img_arr)
    with io.BytesIO() as f:
        img_pil.save(f, format=image_format)
        f.seek(0)
        f.read()
        img_b64 = base64.b64encode(f).decode('utf-8')
    return img_b64.encode('gbk')

def img_b64_to_arr(img_b64):
    '''转换json存储图像数据到np.array'''
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr

def LightCorrection(image: np.ndarray, winsize: int = 512, alpha: float = 0.1, beta: float = 135):
    """光照不均校正"""
    # 边缘填充
    dst_image = np.zeros(shape=(image.shape[0] + winsize, image.shape[1] + winsize), dtype=np.uint8)
    row_start = (dst_image.shape[0] - image.shape[0]) // 2
    col_start = (dst_image.shape[1] - image.shape[1]) // 2

    dst_image[row_start: row_start + image.shape[0], col_start: col_start + image.shape[1]] = image.copy()

    dst_image = dst_image.astype(np.float32)
    # 区域处理
    for row in range(winsize // 2, dst_image.shape[0] - winsize // 2):
        for col in range(winsize // 2, dst_image.shape[1] - winsize // 2):
            roi_image = dst_image[row - winsize // 2: row + winsize // 2 + 1, col - winsize // 2: col + winsize // 2 + 1].copy()
            mu = np.mean(roi_image)
            sigma = np.std(roi_image)
            dst_image[row, col] = (dst_image[row, col] - mu) / (sigma + alpha) * beta + beta

    dst_image[dst_image > 255] = 255
    dst_image[dst_image < 0] = 0
    dst_image = dst_image.astype(np.uint8)

    result_image = dst_image[row_start: row_start + image.shape[0], col_start: col_start + image.shape[1]].copy()

    return result_image

def ReadImage(image_path, mode=0):
    """读取图像"""
    return cv2.imdecode(np.fromfile(image_path, np.uint8), mode)

"""形态学算法实现"""
def HoleFill(src: np.ndarray, connect: int = 4) -> np.ndarray:
    """孔洞填充
    :param src 输入图像（二值图）
    :param connect 连通域类型，4连通或8连通
    :return dst 填充后的图像
    """
    # 模板图像
    template = 255 - src

    # 标记图像
    mark = template.copy()
    mark[1:-1, 1:-1] = 0

    # 形态学重构
    restruct = MorphologicalRestructure(template, mark, connect)

    # 获取结果
    dst = 255 - restruct

    return dst

def MorphologicalRestructure(template: np.ndarray, mark: np.ndarray, connect: int = 4):
    """基于膨胀的形态学重构
    :param template  模板图像
    :param mark  标记图像
    :param connect  连通域类型，4连通或8连通
    :return dst  重构结果
    """
    # 定义结构元素
    kernel_shape = cv2.MORPH_CROSS if connect == 4 else cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(kernel_shape, (3, 3))

    # 膨胀重构
    restruct = mark.copy()
    while True:
        dilate = cv2.dilate(restruct, kernel)
        dilate = cv2.bitwise_and(dilate, template)
        if cv2.countNonZero(cv2.bitwise_xor(restruct, dilate)) == 0:
            break
        restruct = dilate.copy()

    return restruct

def RemoveEdgeObjects(src: np.ndarray, connect: int = 4):
    """移除边界对象
    :param src  输入图像（二值图）
    :param connect 连通域类型，4连通或8连通
    :return dst 移除边界后的图像
    """
    # 模板图像
    template = src.copy()

    # 标记图像
    mark = template.copy()
    mark[1:-1, 1:-1] = 0

    # 形态学重建
    restruct = MorphologicalRestructure(template, mark, connect)

    # 模板图像 - 重构结果
    dst = template - restruct

    return dst

def ConnectedComponents(src: np.ndarray, connect: int = 4):
    """连通分量提取
    :param src  输入的二值图像
    :param connect  连通域类型，4连通或8连通
    :returns labels, component_count 连通域标记图，连通域数量
    """
    # 初始化结果
    labels = np.zeros_like(src, dtype=np.uint16)
    component_count = 0

    # 初始化模板图像
    template = src.copy()

    # 查找前景点
    for row in range(template.shape[0]):
        for col in range(template.shape[1]):
            if template[row, col] == 0:
                continue
            # 标记图像
            mark = np.zeros_like(template)
            mark[row, col] = 255

            # 形态学重建
            restruct = MorphologicalRestructure(template, mark, connect)

            # 更新模板图像
            template -= restruct

            # 更新连通域数量和标记图
            component_count += 1
            labels += (restruct // 255) * component_count

    return labels, component_count

def EdgeAbstract(src: np.ndarray, ksize: tuple = (3, 3)):
    """边界提取
    :param src  输入图像（二值图）
    :param ksize  结构元素尺寸
    :return dst  边界提取结果
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    dst = src - cv2.erode(src, kernel)
    return dst

def MorphThin(binary_image):
    """形态学细化"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    B1 = kernel.copy()
    B1[0, :] = 0
    B1[1, 0] = 0
    B1[1, 2] = 0
    B1C = 1 - B1
    B1C[1, 0] = 0
    B1C[1, 2] = 0

    B2 = kernel.copy()
    B2[0, :] = 0
    B2[:, 2] = 0
    B2C = 1 - B2
    B2C[0, 0] = 0
    B2C[2, 2] = 0

    B = [B1, B2]
    BC = [B1C, B2C]
    for i in range(3):
        B1 = cv2.rotate(B[-2], cv2.ROTATE_90_CLOCKWISE)
        B2 = cv2.rotate(B[-1], cv2.ROTATE_90_CLOCKWISE)
        BC1 = cv2.rotate(BC[-2], cv2.ROTATE_90_CLOCKWISE)
        BC2 = cv2.rotate(BC[-1], cv2.ROTATE_90_CLOCKWISE)
        B.extend([B1, B2])
        BC.extend([BC1, BC2])

    A = binary_image.copy()
    A[A != 0] = 1
    A_pre = A.copy()
    while True:
        for Bi, BiC in zip(B, BC):
            A_pre = A.copy()
            erode1 = cv2.erode(A, Bi)
            erode2 = cv2.erode(1 - A, BiC)
            inter = cv2.bitwise_and(erode1, erode2)
            # BaiseToolFunc.ShowImage(inter * 255)
            A = cv2.bitwise_and(A, 1 - inter)
            if cv2.countNonZero(A_pre - A) == 0:
                break
        if cv2.countNonZero(A_pre - A) == 0:
            break
    return A

def MorphSkeletonize(binary_image):
    """形态学骨架提取"""
    binary_image[binary_image == 255] = 1
    skeleton = np.zeros_like(binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_image = binary_image.copy()
    while True:
        erode = cv2.erode(temp_image, kernel)
        dilate = cv2.dilate(erode, kernel)
        skeleton1 = temp_image - dilate
        skeleton = cv2.bitwise_or(skeleton1, skeleton)
        if cv2.countNonZero(temp_image) == 0:
            break
        temp_image = erode.copy()

    return skeleton

"""图像增强"""
def LogTransform(image: np.ndarray, light_min: int = 50, light_range: int = 200):
    """对数变换"""
    roi = CropImage(image, dsize=(int(image.shape[0] * 0.75), int(image.shape[1] * 0.75)))
    dst = image.astype(np.float32)
    dst = (dst - roi.min()) / (dst.max() - roi.min())
    dst = np.log(dst + 1)
    mask = dst < 0
    dst[mask] = 0
    dst = (dst - dst.min()) / (dst.max() - dst.min()) * light_range
    mask = (1 - mask.astype(np.uint8)) * light_min
    dst += mask
    dst[dst > 250] = 250
    dst = dst.astype(np.uint8)
    return dst

def SaveImage(image: np.ndarray, save_path: str):
    """保存图像"""
    temp_path = "temp." + os.path.basename(save_path).split('.')[-1]
    cv2.imwrite(temp_path, image)
    shutil.move(temp_path, save_path)

def time_metric(func):
    """时间测量装饰器"""
    def wrapper(*args, **kwargs):
        start_time = cv2.getTickCount()
        result = func(*args, **kwargs)
        print(func.__name__ + " lost time: %d ms" % ((cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000))
        return result
    return wrapper

def GaussionBlur(data: list, ksize: int = 3, sigma: float = 2.5):
    """高斯滤波"""
    input_data = np.array(data)
    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = kernel / kernel.sum()
    kernel = np.squeeze(kernel)
    radius = ksize // 2
    dst_data = input_data.copy()
    for i in range(radius, len(data) - radius):
        dst_data[i] = np.sum(input_data[i - radius:i + radius + 1] * kernel)
    return dst_data.tolist()

def ReadYOLOAnnotation(txt_file: str):
    """读取yolo标注文件: box = (cls_id, cx, cy, w, h)"""
    with open(txt_file, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
    bboxes = []
    for line in lines:
        cls_id, cx, cy, w, h = line.strip().split(' ')
        bboxes.append([int(cls_id), float(cx), float(cy), float(w), float(h)])
    return bboxes

def FormatToSquare(src: np.ndarray, is_center: bool = False):
    """填图像到正方形"""
    height, width = src.shape[0], src.shape[1]
    max_length = max(height, width)
    offset_row, offset_col = 0, 0
    if is_center == True:
        offset_col = (max_length - width) // 2
        offset_row = (max_length - height) // 2
        
    if src.ndim == 2:
        shape=(max_length, max_length)
    else:
        shape = (max_length, max_length, src.shape[-1])
    dst = np.zeros(shape=shape, dtype=np.uint8)
    dst[offset_row: offset_row + height, offset_col: offset_col + width] = src.copy()
    return dst

def GammaTransform(image: np.ndarray, gamma: float = 0.5):
    """伽马变换"""
    gamma_table = np.array(range(0, 256, 1), np.float32)
    gamma_table = np.power(gamma_table / 255, gamma) * 255
    gamma_table = gamma_table.astype(np.uint8)
    dst = cv2.LUT(image, gamma_table)
    return dst