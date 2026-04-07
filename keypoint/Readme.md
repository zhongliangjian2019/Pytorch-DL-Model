#### 关键点检模型训练教程
##### 1.目录说明：   
- checkpoints: 保存预训练模型  
- data: 保存训练数据  
- export: 保存导出的onnx模型  
- src: 项目源码
  - train.py: 用于训练模型
  - model.py: 以model开头的相关文件.py定义了模型实现的相关代码
  - dataset.py: 训练数据加载实现代码
  - data_augment.py: 训练过程数据增强实现
  - decode_keypoint.py: 从模型数据解码关键点的实现
  - export_to_onnx.py: 导出模型到onnx
  - json_to_txt.py: 将labelme的json标注转换为yolo式的txt标注文件格式
  - loss_func.py: 定义了用于模型训练的损失函数
  - tool_func.py：定义了一些用于图像操作的工具函数
  - wandb_show.py: 定义了wandb可视化的工具函数
##### 2.模型训练流程
- step-1：准备好数据集, 参见data目录下的数据形式进行组织
- step-2：在train.py完成相关训练配置后运行train.py开启训练, 也可通过命令行配置相关参数
```shell
conda activate py37
python train.py
```
##### 3.其他事项
- 环境配置：opencv / pytorch / pillow / wandb
 