"""
@brief 模型测试
"""
import copy
import time

from models.data_augment import normalize_data
from pointnet import PointNetSegmenter
from pointnet2_seg import PointNet2SegMSG
import numpy as np
import os
import open3d as o3d
import torch
from tqdm import tqdm
from models.pointnet2_utils import farthest_point_sample, index_points

def st3d_dataset_test():
    # 模型定义
    # model = PointNetSegmenter(point_dims=3, input_dims=9, class_num=14, feat_scale=32)
    model = PointNet2SegMSG(point_dims=3, feat_dims=3, num_classes=2)
    model.load_state_dict(torch.load("../ckpts/best.pth", weights_only=True))
    model.eval()
    # 数据准备
    data_dir = "Y:\model_train\dnn_pcd\datasets\chicken3d\datas"
    data_file = os.path.join(os.path.dirname(data_dir), "val.txt")
    with open(data_file, "r") as file:
        filenames = [name.strip('\n') for name in file.readlines()]

    iou_score = 0
    with tqdm(total=len(filenames), desc="running") as pbar:
        for name in filenames:
            data_path = os.path.join(data_dir, name)
            datas = np.loadtxt(data_path, delimiter=' ')
            raw_points = datas[:, :3]
            coord_min, coord_max = np.min(raw_points, axis=0), np.max(raw_points, axis=0)
            stride = 1.0
            grid_x = int((coord_max[0] - coord_min[0]) / stride + 1)
            grid_y = int((coord_max[1] - coord_min[1]) / stride + 1)
            block_list = []
            for x_cell in range(grid_x):
                for y_cell in range(grid_y):
                    xmin = coord_min[0] + x_cell * stride
                    xmax = xmin + stride
                    ymin = coord_min[1] + y_cell * stride
                    ymax = ymin + stride
                    ids = np.where((raw_points[:, 0] >= xmin) & (raw_points[:, 0] < xmax) &
                                   (raw_points[:, 1] >= ymin) & (raw_points[:, 1] < ymax))[0]
                    if len(ids) == 0:
                        continue
                    data = datas[ids, :]
                    points, labels = data[:, :6], data[:, 6:]
                    cx = (xmax + xmin) / 2
                    cy = (ymax + ymin) / 2
                    points[:, 0] -= cx
                    points[:, 1] -= cy
                    points[:, 3:] /= 255
                    relative_points = points[:, :3].copy()
                    relative_points /= coord_max
                    new_data = np.concatenate([points, relative_points, labels], axis=-1)
                    block_list.append(new_data)

            # 模型推理
            block_iou = 0
            for i, block in enumerate(block_list):
                points, labels = block[:, :9], block[:, -1]
                input = torch.as_tensor(points, dtype=torch.float32)
                input = input[None, ...]
                output, _ = model(input)
                pred_labels = torch.argmax(output, dim=-1)[0]
                labels = torch.as_tensor(labels, dtype=torch.int64)
                labels = torch.nn.functional.one_hot(labels, 14).numpy()
                pred_labels = torch.nn.functional.one_hot(pred_labels, 14).numpy()
                inter_set = np.sum(labels * pred_labels)
                union_set = np.sum(labels + pred_labels) - inter_set
                block_iou += (inter_set + 1e-8) / (union_set + 1e-8)
            block_iou /= len(block_list)
            pbar.set_postfix({name.split('.')[0]: "%.4f" % (block_iou)})
            iou_score += block_iou
            pbar.update()
    iou_score /= len(filenames)
    print("iou_score: %.4f" % iou_score)


if __name__ == "__main__":
    # 模型定义
    # model = PointNetSegmenter(point_dims=3, input_dims=9, class_num=14, feat_scale=32)
    model = PointNet2SegMSG(point_dims=3, feat_dims=3, num_classes=2)
    model.load_state_dict(torch.load("../ckpts/pointnet2/last.pth", weights_only=True))
    model.to(device='cuda')
    model.eval()
    # 数据准备
    data_dir = "Y:\model_train\dnn_pcd\datasets\chicken3d\datas"
    data_file = os.path.join(os.path.dirname(data_dir), "val.txt")
    with open(data_file, "r") as file:
        filenames = [name.strip('\n') for name in file.readlines()]

    iou_score = 0
    with tqdm(total=len(filenames), desc="running") as pbar:
        for name in filenames:
            data_path = os.path.join(data_dir, name)
            datas = np.loadtxt(data_path, delimiter=' ', skiprows=1)
            raw_points = datas[:, :3]
            norm_points = normalize_data(raw_points)
            rgb_points = datas[:, 3:6] / 255.0
            labels = datas[:, -1].reshape(-1, 1)
            xyz_rgb = np.concatenate([norm_points, rgb_points, labels], axis=1)
            xyz_rgb = torch.as_tensor(xyz_rgb, dtype=torch.float32).unsqueeze(dim=0)
            ids = farthest_point_sample(xyz_rgb[:, :, :3], 2048)
            new_points = index_points(xyz_rgb, ids)

            # 模型推理
            input = new_points[:, :, :6]
            input = input.to(device="cuda")
            output = model(input)
            output = output.cpu()
            pred_labels = torch.argmax(output, dim=-1)[0]
            pred_marks = pred_labels.cpu().numpy()
            labels = new_points[:, :, -1].to(dtype=torch.int64)
            labels = torch.nn.functional.one_hot(labels, 2).numpy()
            pred_labels = torch.nn.functional.one_hot(pred_labels, 2).numpy()
            inter_set = np.sum(labels * pred_labels)
            union_set = np.sum(labels + pred_labels) - inter_set
            iou = (inter_set + 1e-8) / (union_set + 1e-8)
            pbar.set_postfix({name.split('.')[0]: "%.4f" % (iou)})
            iou_score += iou
            print(np.sum(pred_marks))

            # 分割锚点
            ids = ids.cpu().numpy().reshape(-1)
            anchors = datas[ids, :3]
            anchors = anchors[pred_marks == 1, :]

            # 可视化
            colors = rgb_points
            points = raw_points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 半径检索
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            all_idx = o3d.utility.IntVector()
            for i in range(anchors.shape[0]):
                k, idx, _ = pcd_tree.search_radius_vector_3d(anchors[i, :], radius=0.03)
                all_idx.extend(idx)
            pcd_forward = pcd.select_by_index(all_idx, invert=False)
            pcd_background = pcd.select_by_index(all_idx, invert=True)
            obb = pcd_forward.get_oriented_bounding_box()
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=(0, 0, 0))
            R1 = frame.get_rotation_matrix_from_zyx([0, -np.pi/2, 0])
            R = obb.R @ R1
            frame.rotate(R, center=frame.get_center())
            frame.translate(obb.center, relative=False)
            # o3d.visualization.draw_geometries([pcd_forward, pcd_background, frame])

            pcd_forward_sem = copy.deepcopy(pcd_forward)
            pcd_forward_sem.translate((1.0, 0.0, 0.5), relative=True)
            pcd_forward_sem.paint_uniform_color([1, 0, 0])
            pcd_background_sem = copy.deepcopy(pcd_background)
            pcd_background_sem.translate((1.0, 0.0, 0.5), relative=True)
            pcd_background_sem.paint_uniform_color([0, 0, 1])

            o3d.visualization.draw_geometries([pcd_forward, pcd_background, frame, pcd_forward_sem, pcd_background_sem])
            pbar.update()
    iou_score /= len(filenames)
    print("iou_score: %.4f" % iou_score)




