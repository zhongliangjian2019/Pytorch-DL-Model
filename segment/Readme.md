### 分割模型训练脚本
#### 1. 目录与文件说明  
- ckpts: 保存训练检查点参数, 每次训练会以 “ckpt-日期+时间“ 格式创建目录, 保存当前训练数据.
- data: 数据集目录
  - images: 图像数据, 内含原始图像数据
  - masks: 掩膜数据, 与图像数据对应, 掩膜采用 "p编码“ 格式.
- src: 训练脚本目录, 内含模型训练相关 python 脚本
  - dataset.py: 数据集加载
  - dataset_augment.py: 数据增强
  - dataset_split.py: 数据集划分
  - label2mask.py: labelme 标注数据转 "p编码“ 格式 mask 文件
  - tool_func.py: 工具函数
  - model_loss.py: 分割模型训练损失函数
  - model_module.py: 分割模型定义基础模块
  - model_zoo.py: 分割模型定义, 包含 “unet, pp-liteseg" 等模型
  - model_train.py: 模型训练
  - model_test.py: 模型测试
#### 2. 使用说明
- Step-1: 采用 “labelme” 标注数据, 并采用 "src/label2mask.py" 转换标注到 mask.
  - (注：使用其他标注工具标注数据, 需自行转换)
- Step-2: 采用 “src/dataset_split.py” 进行数据集划分, 用于训练与验证.
- Step-3: 采用 “src/model_train.py" 进行模型训练.
- Step-4: 采用 “src/model_test.py" 进行模型推理测试.