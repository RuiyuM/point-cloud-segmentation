import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import torch_scatter
import torchsparse
import torchsparse.nn.functional
from apricot import CustomSelection, FacilityLocationSelection, FeatureBasedSelection, MaxCoverageSelection
from sklearn.cluster import KMeans
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
from torch.nn.utils import clip_grad_norm_
from sklearn.mixture import GaussianMixture
import copy
from scipy.spatial import cKDTree
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.cluster import DBSCAN


def torch_unique(x):
    unique, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = torch_scatter.scatter_min(perm, inverse, dim=0)[0]
    return unique, inds, inverse, counts


def get_point_in_voxel(raw_coord, voxel_size=0.25, max_points_per_voxel=100, filter_ratio=0.01):
    voxel_grid = (raw_coord / voxel_size).int()
    hash_tensor = torch.cat((voxel_grid, torch.zeros((voxel_grid.shape[0], 1), device=voxel_grid.device)), dim=1).int()
    pc_hash = torchsparse.nn.functional.sphash(hash_tensor)
    sparse_hash, voxel_idx, inverse, voxel_point_counts = torch_unique(pc_hash)
    voxelized_coordinates = voxel_grid[voxel_idx]

    inverse_sorted, sorted_to_inverse_index = torch.sort(inverse)
    index = torch.arange(inverse_sorted.shape[0], device=raw_coord.device)
    first_locate_index = torch_scatter.scatter_min(index, inverse_sorted, dim=0)[0]

    k = int(voxelized_coordinates.shape[0] * filter_ratio)
    max_voxel_point = min(torch.topk(voxel_point_counts, k=k, largest=True)[0][-1], max_points_per_voxel)

    temp = first_locate_index[:, None] - torch.zeros(max_voxel_point, device=raw_coord.device, dtype=torch.int64)
    sorted_locate_index = temp + torch.arange(0, max_voxel_point, device=raw_coord.device)
    point_index_bound = torch.cat((temp, torch.full([1, max_voxel_point], raw_coord.shape[0], device=raw_coord.device)),
                                  dim=0)[1:]
    sorted_to_inverse_index_temp = torch.cat((sorted_to_inverse_index, torch.tensor([-1], device=raw_coord.device)),
                                             dim=0)
    point_in_voxel = sorted_to_inverse_index_temp[
        torch.where(point_index_bound > sorted_locate_index, sorted_locate_index, -1)]

    point_in_voxel[inverse[point_in_voxel[:, 0]]] = point_in_voxel.clone()

    voxel_point_counts = torch.clamp(voxel_point_counts, 0, max_voxel_point)
    return voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts


class MILANVoxelSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 voxel_size: float = 0.25,
                 self_sup_weight: float = 0.8,
                 cluster_ratio: float = 0.001,
                 max_points_per_voxel: int = 100,
                 device: str = "cuda"):
        self.select_num = select_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.self_sup_weight = self_sup_weight  # 自监督特征权重
        self.cluster_ratio = cluster_ratio  # 标注比例(0.1%)
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device

        # 自监督特征记忆
        self.selected_scans = []
        self.self_sup_features = []

    def _get_self_supervised_feature(self, raw_coord, preds):
        """融合几何与语义的自监督特征"""
        # 几何特征：体素中心坐标
        geo_feat = raw_coord.mean(dim=0)

        # 语义特征：预测分布熵
        prob = F.softmax(preds, dim=-1)
        sem_feat = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1).mean()

        return torch.cat([geo_feat, sem_feat.unsqueeze(0)])

    def _select_informative_scans(self, current_feature):
        """基于自监督特征的扫描选择"""
        if not self.selected_scans:
            return True

        # 计算与历史扫描的相似度
        similarities = F.cosine_similarity(current_feature.unsqueeze(0),
                                           torch.stack(self.self_sup_features))
        return torch.max(similarities) < 0.7  # 相似度阈值

    def _cluster_annotation(self, points, preds):
        """聚类标注模拟"""
        # 特征提取：坐标+预测概率
        features = torch.cat([points, F.softmax(preds, dim=-1)], dim=-1).cpu().numpy()

        # 自动确定聚类数量
        n_clusters = max(1, int(len(points) * self.cluster_ratio))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)

        # 为每个簇生成伪标签
        cluster_labels = []
        for i in range(n_clusters):
            cluster_points = points[kmeans.labels_ == i]
            cluster_preds = preds[kmeans.labels_ == i]
            dominant_class = cluster_preds.argmax(dim=-1).mode().values
            cluster_labels.append(dominant_class)

        return cluster_labels

    def select(self, mask, raw_coord, preds, true_label=None):
        # Step 1: 提取自监督特征
        current_feature = self._get_self_supervised_feature(raw_coord, preds)

        # Step 2: 扫描选择策略
        if self._select_informative_scans(current_feature):
            self.selected_scans.append(current_feature)
            self.self_sup_features.append(current_feature)

            # Step 3: 聚类标注
            cluster_labels = self._cluster_annotation(raw_coord, preds)

            # Step 4: 更新mask(模拟点击标注)
            annotated_points = []
            for label in cluster_labels:
                # 选择离聚类中心最近的3个点作为标注点
                center = raw_coord.mean(dim=0)
                distances = torch.norm(raw_coord - center, dim=1)
                topk_indices = torch.topk(distances, k=3, largest=False).indices
                annotated_points.extend(topk_indices.tolist())

            mask[annotated_points] = True

        # Step 5: 生成统计信息(与ReDAL兼容)
        point_cloud_stat_true = []
        point_cloud_stat_pre = []
        if true_label is not None and len(cluster_labels) > 0:
            for label in cluster_labels:
                # 假设每个簇的标注传播到所有成员
                cluster_mask = (true_label == label)
                true_counts = torch.histc(true_label[cluster_mask].float(),
                                          bins=self.num_classes, min=0, max=self.num_classes - 1)
                pred_counts = torch.histc(preds.argmax(dim=-1)[cluster_mask].float(),
                                          bins=self.num_classes, min=0, max=self.num_classes - 1)

                point_cloud_stat_true.append((true_counts.unsqueeze(0), cluster_mask.sum().item()))
                point_cloud_stat_pre.append((pred_counts.unsqueeze(0), cluster_mask.sum().item()))

        return mask, point_cloud_stat_true, point_cloud_stat_pre


class MarginSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None):
        if self.select_method == 'voxel':
            # Voxelize the point cloud
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            # Get predicted labels
            preds_label = preds.max(dim=-1).indices

            # Map predicted and true labels to voxels
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            # Calculate class distribution within each voxel
            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            # Calculate class probability and entropy (confusion)
            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            # Sort voxels by confusion (entropy) and select the most uncertain ones
            _, voxel_indices = torch.sort(confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False

            # Calculate true and predicted class distributions for selected voxels
            point_cloud_stat_true = []
            point_cloud_stat_pre = []
            pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                             dtype=torch.int)
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_true[:, i] = class_num_count_i
                class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_predicted[:, i] = class_num_count_i
            point_cloud_stat_true.append((pre_selected_label_count_true, voxel_point_counts[voxel_indices[0]]))
            point_cloud_stat_pre.append((pre_selected_label_count_predicted, voxel_point_counts[voxel_indices[0]]))

            return mask, point_cloud_stat_true, point_cloud_stat_pre
        else:
            raise NotImplementedError


class BaSALVoxelSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 voxel_size: float = 0.25,
                 warm_start_steps: int = 100,
                 size_balance_weight: float = 0.5,
                 max_points_per_voxel: int = 100,
                 device: str = "cuda"):
        self.select_num = select_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.warm_start_steps = warm_start_steps
        self.size_balance_weight = size_balance_weight
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device

        # 尺寸平衡相关
        self.class_weights = torch.ones(num_classes, device=device)  # 类别权重
        self.class_counts = torch.zeros(num_classes, device=device)  # 已选类别统计
        self.step_counter = 0  # 总选择次数计数器

        # 预热阶段相关
        self.warm_start_cache = []  # 存储预热阶段几何特征
        self.coverage_map = None  # 空间覆盖地图

    def _calculate_coverage_score(self, voxel_coords):
        """计算空间覆盖分数（预热阶段使用）"""
        if self.coverage_map is None:
            self.coverage_map = torch.zeros_like(voxel_coords.float())

        # 计算与已覆盖区域的曼哈顿距离
        dist = torch.cdist(voxel_coords.float(), self.coverage_map, p=1)
        min_dist = torch.min(dist, dim=1)[0]
        return min_dist / (self.voxel_size * 10)  # 归一化

    def _update_class_weights(self, selected_labels):
        """动态更新类别平衡权重"""
        unique, counts = torch.unique(selected_labels, return_counts=True)
        self.class_counts[unique] += counts.float()

        # 计算逆向频率权重
        total = self.class_counts.sum()
        freq = self.class_counts / (total + 1e-6)
        self.class_weights = 1.0 / (freq + 0.1)  # 平滑处理

    def _get_voxel_class_weight(self, voxel_labels):
        """计算voxel的类别平衡得分"""
        class_dist = torch.histc(voxel_labels.float(),
                                 bins=self.num_classes,
                                 min=0,
                                 max=self.num_classes - 1)
        return torch.sum(class_dist * self.class_weights) / len(voxel_labels)

    def select(self, mask, raw_coord, preds, true_label=None):
        # 保持与ReDAL完全相同的输入参数
        voxel_coords, _, _, point_in_voxel, _ = get_point_in_voxel(
            raw_coord, self.voxel_size, self.max_points_per_voxel)

        # 第一阶段：预热阶段（基于空间覆盖）
        if self.step_counter < self.warm_start_steps:
            # 计算覆盖分数
            coverage_scores = self._calculate_coverage_score(voxel_coords)

            # 选择策略：覆盖分数 + 随机采样
            _, topk_indices = torch.topk(coverage_scores,
                                         min(self.select_num, len(coverage_scores)))
            selected_voxels = topk_indices.tolist()

            # 更新覆盖地图
            self.coverage_map = torch.cat([self.coverage_map,
                                           voxel_coords[selected_voxels].float()])
        else:
            # 第二阶段：正式阶段（不确定性+尺寸平衡）
            voxel_scores = []
            valid_voxels = []

            for voxel_id, voxel_points in enumerate(point_in_voxel):
                if len(voxel_points) == 0:
                    continue
                valid_voxels.append(voxel_id)

                # 不确定性计算（与ReDAL一致）
                voxel_preds = preds[voxel_points]
                prob = F.softmax(voxel_preds, dim=-1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1).mean()

                # 尺寸平衡得分
                if true_label is not None:
                    voxel_labels = true_label[voxel_points]
                else:
                    voxel_labels = voxel_preds.argmax(dim=-1)

                balance_score = self._get_voxel_class_weight(voxel_labels)

                # 综合得分
                combined_score = entropy + self.size_balance_weight * balance_score
                voxel_scores.append(combined_score)

            # 选择Top-K voxels
            scores_tensor = torch.stack(voxel_scores)
            _, topk_indices = torch.topk(scores_tensor,
                                         min(self.select_num, len(scores_tensor)))
            selected_voxels = [valid_voxels[i] for i in topk_indices.cpu().numpy()]

            # 更新类别权重
            if true_label is not None:
                selected_labels = torch.cat([true_label[point_in_voxel[v]] for v in selected_voxels])
                self._update_class_weights(selected_labels)

        # 更新mask
        for voxel_id in selected_voxels:
            mask[point_in_voxel[voxel_id]] = True

        # 统计信息生成（与ReDAL完全一致）
        point_cloud_stat_true = []
        point_cloud_stat_pre = []
        if true_label is not None:
            for voxel_id in selected_voxels:
                voxel_points = point_in_voxel[voxel_id]

                # 真实标签统计
                true_counts = torch.histc(true_label[voxel_points].float(),
                                          bins=self.num_classes,
                                          min=0,
                                          max=self.num_classes - 1)

                # 预测统计
                pred_labels = preds[voxel_points].argmax(dim=-1)
                pred_counts = torch.histc(pred_labels.float(),
                                          bins=self.num_classes,
                                          min=0,
                                          max=self.num_classes - 1)

                point_cloud_stat_true.append((true_counts.unsqueeze(0), len(voxel_points)))
                point_cloud_stat_pre.append((pred_counts.unsqueeze(0), len(voxel_points)))

        self.step_counter += 1
        return mask, point_cloud_stat_true, point_cloud_stat_pre

class CoreSetSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 13,  # Assuming 13 valid classes + 1 ignore
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'mean'):
        self.select_num = select_num
        self.num_classes = num_classes
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.voxel_select_method = voxel_select_method

    def select(self, mask, raw_coord, preds, true_label):
        true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=mask.device, dtype=torch.int32)
        point_cloud_stat_true = []
        point_cloud_stat_pre = []

        if self.select_method == 'voxel':
            # Voxelization
            voxelized_coord, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            # Get predictions and labels
            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            # Calculate class distributions
            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=mask.device, dtype=torch.int32)
            for i in range(1, self.num_classes + 1):
                class_num_count[:, i] = torch.sum((point_in_voxel_label == i).int(), dim=1)

            # CoreSet selection (Facility Location)
            valid_voxels_mask = (voxel_point_counts > 0)
            valid_class_counts = class_num_count[valid_voxels_mask]

            # Normalize counts to probabilities
            class_probs = valid_class_counts.float() / voxel_point_counts[valid_voxels_mask].unsqueeze(1)

            # Compute pairwise similarity matrix
            similarity_matrix = torch.mm(class_probs, class_probs.T)

            # Greedy facility location selection
            selected_indices = []
            for _ in range(self.select_num):
                if len(selected_indices) == 0:
                    scores = similarity_matrix.sum(dim=1)
                else:
                    current_sim = similarity_matrix[:, selected_indices]
                    scores = (similarity_matrix.max(dim=1).values - current_sim.max(dim=1).values).clamp(min=0)

                new_idx = scores.argmax().item()
                selected_indices.append(new_idx)

            # Convert to original voxel indices
            valid_voxel_indices = torch.where(valid_voxels_mask)[0]
            selected_voxels = valid_voxel_indices[selected_indices]

            # Update mask and collect statistics
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)
            for voxel_id in selected_voxels:
                points_in_voxel = point_in_voxel[voxel_id]
                valid_points = points_in_voxel[points_in_voxel != -1]
                new_points = valid_points[~mask[valid_points]]

                if len(new_points) > 0:
                    mask[new_points] = True

                    # True label counts
                    true_labels = true_label[new_points].clamp(0, self.num_classes)
                    unique_true, counts_true = torch.unique(true_labels, return_counts=True)
                    true_counts = torch.zeros(self.num_classes + 1, dtype=torch.int32, device=mask.device)
                    true_counts[unique_true] = counts_true.int()

                    # Predicted label counts
                    pred_labels = preds_label[new_points].clamp(0, self.num_classes - 1)
                    unique_pred, counts_pred = torch.unique(pred_labels, return_counts=True)
                    pred_counts = torch.zeros(self.num_classes + 1, dtype=torch.int32, device=mask.device)
                    pred_counts[unique_pred] = counts_pred.int()

                    # Append statistics
                    point_cloud_stat_true.append((true_counts.unsqueeze(0), len(new_points)))
                    point_cloud_stat_pre.append((pred_counts.unsqueeze(0), len(new_points)))
                    true_selected_label_count += true_counts.unsqueeze(0)

            # Maintain original return format
            final_true_stat = [point_cloud_stat_true[0]] if point_cloud_stat_true else []
            final_pre_stat = [point_cloud_stat_pre[0]] if point_cloud_stat_pre else []

            return mask, final_true_stat, final_pre_stat

        else:
            raise NotImplementedError


class LiDALVoxelSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 voxel_size: float = 0.25,
                 temporal_window: int = 5,
                 uncertainty_weight: float = 0.7,
                 consistency_weight: float = 0.3,
                 max_points_per_voxel: int = 100,
                 device: str = "cuda"):
        self.select_num = select_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.temporal_window = temporal_window
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device

        # 时空记忆组件
        self.history = {
            'features': [],
            'coords': [],
            'preds': []
        }

    def _spatial_alignment(self, current_coords):
        """基于几何坐标的时空对齐（无需pose信息）"""
        if not self.history['coords']:
            return []

        # 使用最近邻匹配历史voxel
        hist_coords = torch.cat(self.history['coords'])
        dist_matrix = torch.cdist(current_coords, hist_coords)
        return dist_matrix.argmin(dim=1)

    def _temporal_uncertainty(self, current_pred, matched_hist_preds):
        """计算时间维度不确定性"""
        if not matched_hist_preds:
            return torch.tensor(0.0, device=self.device)

        current_prob = F.softmax(current_pred, dim=0)
        hist_probs = [F.softmax(p, dim=0) for p in matched_hist_preds]

        # 计算平均KL散度
        kl_sum = sum(F.kl_div(current_prob.log(), hp, reduction='sum') for hp in hist_probs)
        return kl_sum / len(hist_probs)

    def select(self, mask, raw_coord, preds, true_label=None):
        # 保持与原始ReDAL完全相同的输入参数
        _, _, _, point_in_voxel, voxel_coords = get_point_in_voxel(
            raw_coord, self.voxel_size, self.max_points_per_voxel)

        # 当前帧特征提取
        valid_voxels = []
        current_features = []
        for voxel_id, voxel_points in enumerate(point_in_voxel):
            if len(voxel_points) == 0:
                continue
            valid_voxels.append(voxel_id)

            # 几何特征（与ReDAL保持一致）
            geo_center = raw_coord[voxel_points].mean(dim=0)
            pred_dist = F.softmax(preds[voxel_points], dim=-1).mean(dim=0)
            current_features.append(torch.cat([geo_center, pred_dist]))

        # 时空对齐
        current_coords = voxel_coords[valid_voxels]
        matched_indices = self._spatial_alignment(current_coords)

        # 双维度不确定性计算
        uncertainties = []
        consistencies = []
        for idx, voxel_id in enumerate(valid_voxels):
            voxel_points = point_in_voxel[voxel_id]

            # 帧内不确定性（保持ReDAL计算方式）
            voxel_preds = preds[voxel_points]
            prob = F.softmax(voxel_preds, dim=-1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1).mean()

            # 帧间不确定性
            matched_preds = [self.history['preds'][i] for i in matched_indices[idx]] \
                if idx < len(matched_indices) else []
            temp_uncertainty = self._temporal_uncertainty(prob.mean(dim=0), matched_preds)

            uncertainties.append(entropy)
            consistencies.append(temp_uncertainty)

        # 综合得分计算
        u_tensor = torch.stack(uncertainties)
        c_tensor = torch.stack(consistencies)
        scores = self.uncertainty_weight * u_tensor + self.consistency_weight * c_tensor

        # 选择逻辑（保持ReDAL结构）
        _, topk_indices = torch.topk(scores, min(self.select_num, len(scores)))
        selected_voxels = [valid_voxels[i] for i in topk_indices.cpu().numpy()]

        # 更新mask
        for vid in selected_voxels:
            mask[point_in_voxel[vid]] = True

        # 统计信息生成（与ReDAL完全兼容）
        stat_true, stat_pre = [], []
        if true_label is not None:
            for vid in selected_voxels:
                points = point_in_voxel[vid]

                # 真实标签统计
                true_counts = torch.histc(true_label[points].float(),
                                          bins=self.num_classes, min=0, max=self.num_classes - 1)

                # 预测统计
                pred_labels = preds[points].argmax(dim=-1)
                pred_counts = torch.histc(pred_labels.float(),
                                          bins=self.num_classes, min=0, max=self.num_classes - 1)

                stat_true.append((true_counts.unsqueeze(0), len(points)))
                stat_pre.append((pred_counts.unsqueeze(0), len(points)))

        # 更新时空记忆（滑动窗口）
        self._update_memory(current_features, current_coords, preds)

        return mask, stat_true, stat_pre

    def _update_memory(self, features, coords, preds):
        """记忆更新机制"""
        self.history['features'].append(torch.stack(features))
        self.history['coords'].append(coords)
        self.history['preds'].extend([p.mean(dim=0) for p in preds])

        # 维持时间窗口
        if len(self.history['preds']) > self.temporal_window * 100:  # 假设每帧约100个voxel
            remove_num = len(self.history['preds']) - self.temporal_window * 100
            self.history['features'] = self.history['features'][1:]
            self.history['coords'] = self.history['coords'][1:]
            self.history['preds'] = self.history['preds'][remove_num:]

class ReDALVoxelSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 voxel_size: float = 0.25,
                 diversity_weight: float = 0.5,
                 max_points_per_voxel: int = 100,
                 device: str = "cuda"):
        self.select_num = select_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.diversity_weight = diversity_weight
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device
        self.selected_voxel_features = []  # 存储已选voxel特征

    def _get_voxel_features(self, voxel_points, voxel_preds):
        """生成voxel特征向量：几何中心 + 预测分布"""
        geo_center = voxel_points.mean(dim=0)  # [3]
        pred_dist = F.softmax(voxel_preds, dim=-1).mean(dim=0)  # [num_classes]
        return torch.cat([geo_center, pred_dist])  # [3 + num_classes]

    def select(self, mask, raw_coord, preds, true_label=None):
        # Step 1: Voxel划分
        _, _, _, point_in_voxel, voxel_counts = get_point_in_voxel(
            raw_coord, self.voxel_size, self.max_points_per_voxel)

        # Step 2: 计算每个voxel的特征和不确定性
        voxel_features = []
        voxel_uncertainty = []
        valid_voxels = []

        for voxel_id in range(len(point_in_voxel)):
            voxel_points = point_in_voxel[voxel_id]
            if len(voxel_points) == 0:
                continue
            valid_voxels.append(voxel_id)

            # 计算不确定性（预测熵）
            voxel_preds = preds[voxel_points]
            prob = F.softmax(voxel_preds, dim=-1)
            entropy = -torch.sum(prob * torch.log2(prob + 1e-10), dim=-1).mean()
            voxel_uncertainty.append(entropy.item())

            # 生成特征向量
            feat = self._get_voxel_features(raw_coord[voxel_points], voxel_preds)
            voxel_features.append(feat.cpu().numpy())

        # Step 3: 多样性计算（与已选voxel的相似度）
        diversity_scores = np.zeros(len(valid_voxels))
        if len(self.selected_voxel_features) > 0:
            similarities = cosine_similarity(voxel_features, self.selected_voxel_features)
            diversity_scores = np.max(similarities, axis=1)  # 取最大相似度

        # Step 4: 混合得分计算
        uncertainty_tensor = torch.tensor(voxel_uncertainty, device=self.device)
        diversity_penalty = torch.tensor(diversity_scores, device=self.device) * self.diversity_weight
        hybrid_scores = uncertainty_tensor - diversity_penalty

        # Step 5: 选择Top-K voxels
        _, topk_indices = torch.topk(hybrid_scores, min(self.select_num, len(hybrid_scores)))
        selected_voxel_ids = [valid_voxels[i] for i in topk_indices.cpu().numpy()]

        # Step 6: 更新mask并记录特征
        for voxel_id in selected_voxel_ids:
            points_in_voxel = point_in_voxel[voxel_id]
            mask[points_in_voxel] = True
            self.selected_voxel_features.append(voxel_features[voxel_id])

        # Step 7: 统计信息生成（保持与原始格式兼容）
        point_cloud_stat_true = []
        point_cloud_stat_pre = []

        if true_label is not None:
            for voxel_id in selected_voxel_ids:
                voxel_points = point_in_voxel[voxel_id]

                # 真实标签统计
                true_labels = true_label[voxel_points]
                true_count = torch.histc(true_labels.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

                # 预测标签统计
                pred_labels = preds[voxel_points].argmax(dim=-1)
                pred_count = torch.histc(pred_labels.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

                point_cloud_stat_true.append((true_count.unsqueeze(0), len(voxel_points)))
                point_cloud_stat_pre.append((pred_count.unsqueeze(0), len(voxel_points)))

        return mask, point_cloud_stat_true, point_cloud_stat_pre

class RandomSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100, ):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None):
        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)
            # perm = torch.randperm(voxel_idx.shape[0])
            perm = torch.randperm(voxel_idx.shape[0], device=point_in_voxel.device)
            index = perm[mask[point_in_voxel[perm][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            class_num_count_cpu = point_in_voxel_label_true.cpu()  # Move to CPU
            class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(0, self.num_classes + 1):
                # x = index[:self.select_num]
                class_num_count_i = torch.where(point_in_voxel_label_true[index[:self.select_num][0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                true_selected_label_count[:, i] = class_num_count_i
            return mask, true_selected_label_count
        else:
            raise NotImplementedError

class RandomSelect_collect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100, ):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def count_uniform_label_rows(self, tensor):
        # Initialize a counter to track the number of rows that meet the condition
        uniform_row_count = 0

        # Iterate over each row in the tensor
        for row in tensor:
            # Filter out `-1` values
            filtered_values = row[row != -1]

            # Check if the filtered array has exactly one unique element or is empty
            if filtered_values.unique().size(0) == 1 or filtered_values.nelement() == 0:
                # Increment the counter if the condition is met
                uniform_row_count += 1

        return uniform_row_count

    def select(self, mask, raw_coord=None, preds=None, true_label=None):
        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)
            # perm = torch.randperm(voxel_idx.shape[0])
            perm = torch.randperm(voxel_idx.shape[0], device=point_in_voxel.device)
            index = perm[mask[point_in_voxel[perm][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            count_uniform_voxel = self.count_uniform_label_rows(point_in_voxel_label_true)
            num_voxels, points_per_voxel = point_in_voxel.shape

            count_total_voxel = num_voxels
            for i in range(0, self.num_classes + 1):
                # x = index[:self.select_num]
                class_num_count_i = torch.where(point_in_voxel_label_true[index[:self.select_num][0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                true_selected_label_count[:, i] = class_num_count_i
            return mask, true_selected_label_count, count_uniform_voxel, count_total_voxel
        else:
            raise NotImplementedError


class RandomSelect_collect_point:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100, ):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def count_uniform_label_rows(self, tensor):
        uniform_row_count = 0
        for row in tensor:
            filtered_values = row[row != -1]
            if filtered_values.unique().size(0) <= 1:
                uniform_row_count += 1
        return uniform_row_count

    def select(self, mask, raw_coord=None, preds=None, true_label=None):
        if self.select_method == 'voxel':
            # 原有体素选择逻辑保持不变
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)
            perm = torch.randperm(voxel_idx.shape[0], device=point_in_voxel.device)
            index = perm[mask[point_in_voxel[perm][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            count_uniform_voxel = self.count_uniform_label_rows(point_in_voxel_label_true)
            num_voxels, points_per_voxel = point_in_voxel.shape

            count_total_voxel = num_voxels
            for i in range(0, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label_true[index[:self.select_num][0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                true_selected_label_count[:, i] = class_num_count_i
            return mask, true_selected_label_count, count_uniform_voxel, count_total_voxel
        elif self.select_method == 'point':
            # 直接选择点的新逻辑
            num_points = true_label.size(0)  # 获取总点数
            perm = torch.randperm(num_points, device=mask.device)  # 生成所有点的随机排列
            available_indices = perm[~mask[perm]]  # 筛选未被mask选中的点

            # 选择前select_num个可用点
            selected_indices = available_indices[:self.select_num]

            # 更新mask，标记选中的点
            mask[selected_indices] = True

            # 统计选中点的真实标签分布（将-1映射为num_classes）
            adjusted_labels = true_label[selected_indices].clone()
            adjusted_labels[adjusted_labels == -1] = self.num_classes  # 将-1视为最后一个类别

            # 计算各类别数量
            counts = torch.bincount(adjusted_labels, minlength=self.num_classes + 1)
            true_selected_label_count = counts.unsqueeze(0)  # 保持输出形状为(1, num_classes+1)

            # 在点模式下，以下参数无意义，返回默认值
            count_uniform_voxel = 0  # 点模式下无体素均匀性统计
            count_total_voxel = selected_indices.size(0)  # 实际选中的点数

            return mask, true_selected_label_count, count_uniform_voxel, count_total_voxel
        else:
            raise NotImplementedError(f"Select method '{self.select_method}' not supported.")


class LIMVoxelSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 voxel_size: float = 0.25,
                 uncertainty_thresh: float = 0.8,
                 diversity_ratio: float = 0.3,
                 max_points_per_voxel: int = 100,
                 device: str = "cuda"):
        # 核心参数
        self.select_num = select_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.uncertainty_thresh = uncertainty_thresh  # 不确定性阈值:cite[7]
        self.diversity_ratio = diversity_ratio  # 多样性采样比例:cite[8]
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device

        # 标签推荐状态跟踪
        self.labeled_voxels = set()  # 已标注voxel记录
        self.class_entropy = torch.zeros(num_classes, device=device)  # 各类别不确定性累计
        self.feature_bank = []  # 特征记忆库:cite[5]

    def _compute_semantic_uncertainty(self, preds):
        """混合不确定性计算(预测熵+类别分布方差)"""
        prob = F.softmax(preds, dim=-1)
        # 确保计算的是标量值
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1).mean()  # 平均预测熵
        class_variance = torch.var(prob, dim=0).mean()  # 平均类别分布方差
        return (entropy + class_variance).item()  # 转换为Python float

    def _cluster_diverse_sampling(self, features, k):
        """基于特征聚类的多样性采样:cite[8]"""
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        cluster_centers = kmeans.cluster_centers_
        return [np.argmin(np.linalg.norm(features - center, axis=1)) for center in cluster_centers]

    def select(self, mask, raw_coord, preds, true_label=None):
        # Step 1: Voxel划分(与ReDAL相同)
        _, _, _, point_in_voxel, voxel_coords = get_point_in_voxel(
            raw_coord, self.voxel_size, self.max_points_per_voxel)

        # Step 2: 候选voxel筛选
        candidate_voxels = []
        voxel_features = []
        for voxel_id, voxel_points in enumerate(point_in_voxel):
            if len(voxel_points) == 0 or voxel_id in self.labeled_voxels:
                continue

            # 特征提取(几何+语义)
            geo_center = raw_coord[voxel_points].mean(dim=0)
            sem_feat = preds[voxel_points].mean(dim=0)
            feat = torch.cat([geo_center, sem_feat]).cpu().numpy()

            candidate_voxels.append(voxel_id)
            voxel_features.append(feat)

        # Step 3: 混合推荐策略
        if len(candidate_voxels) == 0:
            return mask, [], []

        # 不确定性筛选(前70%)
        uncertainties = [self._compute_semantic_uncertainty(preds[point_in_voxel[vid]]) for vid in candidate_voxels]
        sorted_idx = np.argsort(uncertainties)[::-1]
        top_uncertain = sorted_idx[:int(len(sorted_idx) * 0.7)]

        # 多样性采样(剩余30%)
        if self.diversity_ratio > 0:
            cluster_num = int(len(top_uncertain) * self.diversity_ratio)
            cluster_features = [voxel_features[i] for i in top_uncertain]
            cluster_samples = self._cluster_diverse_sampling(cluster_features, cluster_num)
            selected_idx = [top_uncertain[i] for i in cluster_samples]
        else:
            selected_idx = top_uncertain[:self.select_num]

        # Step 4: 标签推荐
        final_voxels = [candidate_voxels[i] for i in selected_idx[:self.select_num]]

        # 更新标注状态
        uncertainties = []
        for vid in candidate_voxels:
            voxel_preds = preds[point_in_voxel[vid]]
            # 确保返回标量值
            uncertainty = self._compute_semantic_uncertainty(voxel_preds)
            uncertainties.append(uncertainty)  # 现在这是Python float列表

        # 转换为numpy数组
        uncertainties_np = np.array(uncertainties)
        sorted_idx = np.argsort(uncertainties_np)[::-1]

        # Step 5: 生成统计信息(与ReDAL完全兼容)
        point_cloud_stat_true = []
        point_cloud_stat_pre = []
        if true_label is not None:
            for vid in final_voxels:
                voxel_points = point_in_voxel[vid]

                # 真实标签统计
                true_counts = torch.histc(true_label[voxel_points].float(),
                                          bins=self.num_classes,
                                          min=0,
                                          max=self.num_classes - 1)

                # 预测统计
                pred_labels = preds[voxel_points].argmax(dim=-1)
                pred_counts = torch.histc(pred_labels.float(),
                                          bins=self.num_classes,
                                          min=0,
                                          max=self.num_classes - 1)

                point_cloud_stat_true.append((true_counts.unsqueeze(0), len(voxel_points)))
                point_cloud_stat_pre.append((pred_counts.unsqueeze(0), len(voxel_points)))

        return mask, point_cloud_stat_true, point_cloud_stat_pre


class SSDReconVoxelSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 voxel_size: float = 0.25,
                 structural_weight: float = 0.6,
                 spatial_weight: float = 0.4,
                 curvature_threshold: float = 0.3,
                 max_points_per_voxel: int = 100,
                 device: str = "cuda"):
        self.select_num = select_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.structural_weight = structural_weight  # 结构多样性权重
        self.spatial_weight = spatial_weight  # 空间多样性权重
        self.curvature_threshold = curvature_threshold
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device

        # 空间结构记忆
        self.selected_coords = torch.empty((0, 3), device=device)
        self.selected_features = torch.empty((0, 6), device=device)  # 存储XYZ+法向量

    def _compute_structural_features(self, points):
        """计算结构特征（法向量+曲率）"""
        # 协方差分析
        pts_centered = points - points.mean(dim=0)
        cov = torch.mm(pts_centered.T, pts_centered) / (pts_centered.shape[0] - 1)

        # 特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # 法向量取最小特征值对应向量
        normal = eigenvectors[:, 0]

        # 曲率计算
        curvature = eigenvalues[0] / (eigenvalues.sum() + 1e-6)

        return normal, curvature

    def _spatial_diversity_score(self, coords):
        """空间多样性评分"""
        if self.selected_coords.size(0) == 0:
            return torch.zeros(coords.size(0), device=self.device)

        # 计算与已选区域的最近距离
        dist_matrix = torch.cdist(coords, self.selected_coords)
        min_dist = dist_matrix.min(dim=1)[0]
        return torch.sigmoid(min_dist / self.voxel_size)  # 归一化到[0,1]

    def _structural_diversity_score(self, features):
        """结构多样性评分"""
        if self.selected_features.size(0) == 0:
            return torch.ones(features.size(0), device=self.device)

        # 特征相似度计算
        sim_matrix = F.cosine_similarity(features.unsqueeze(1),
                                         self.selected_features.unsqueeze(0),
                                         dim=-1)
        return 1 - sim_matrix.max(dim=1)[0]  # 取最大相似度的补集

    def select(self, mask, raw_coord, preds, true_label=None):
        # Step 1: Voxel划分（与ReDAL相同）
        _, _, _, point_in_voxel, voxel_coords = get_point_in_voxel(
            raw_coord, self.voxel_size, self.max_points_per_voxel)

        # Step 2: 计算结构特征
        valid_voxels = []
        structural_features = []
        spatial_coords = []

        for voxel_id, voxel_points in enumerate(point_in_voxel):
            if len(voxel_points) == 0:
                continue
            valid_voxels.append(voxel_id)
            points = raw_coord[voxel_points]

            # 计算法向量和曲率
            normal, curvature = self._compute_structural_features(points)

            # 筛选高曲率区域
            if curvature < self.curvature_threshold:
                continue

            # 构建特征向量（坐标+法向量）
            geo_center = points.mean(dim=0)
            feature = torch.cat([geo_center, normal])
            structural_features.append(feature)
            spatial_coords.append(voxel_coords[voxel_id])

        if len(structural_features) == 0:
            return mask, [], []

        # 转换为Tensor
        feature_tensor = torch.stack(structural_features)
        coord_tensor = torch.stack(spatial_coords)

        # Step 3: 多样性评分计算
        spatial_scores = self._spatial_diversity_score(coord_tensor)
        structural_scores = self._structural_diversity_score(feature_tensor)

        # 综合评分
        combined_scores = (self.spatial_weight * spatial_scores +
                           self.structural_weight * structural_scores)

        # Step 4: 密度聚类采样
        features_np = feature_tensor.cpu().numpy()
        dbscan = DBSCAN(eps=0.5, min_samples=3).fit(features_np)
        cluster_labels = dbscan.labels_

        # 按聚类选择代表点
        selected = []
        for label in np.unique(cluster_labels):
            if label == -1:  # 噪声点
                continue
            mask = cluster_labels == label
            cluster_scores = combined_scores[mask]
            selected.append(valid_voxels[mask][torch.argmax(cluster_scores).item()])

        # Step 5: 补充选择（不足select_num时）
        remaining = self.select_num - len(selected)
        if remaining > 0:
            top_scores, top_indices = torch.topk(combined_scores, remaining)
            selected += [valid_voxels[i] for i in top_indices.cpu().numpy()]

        # 更新mask和记忆
        for vid in selected:
            points = point_in_voxel[vid]
            mask[points] = True
            self.selected_coords = torch.cat([self.selected_coords,
                                              voxel_coords[vid].unsqueeze(0)])
            self.selected_features = torch.cat([self.selected_features,
                                                feature_tensor[vid].unsqueeze(0)])

        # 统计信息生成（与ReDAL完全兼容）
        point_cloud_stat_true = []
        point_cloud_stat_pre = []
        if true_label is not None:
            for vid in selected:
                voxel_points = point_in_voxel[vid]

                # 真实标签统计
                true_counts = torch.histc(true_label[voxel_points].float(),
                                          bins=self.num_classes, min=0, max=self.num_classes - 1)

                # 预测统计
                pred_labels = preds[voxel_points].argmax(dim=-1)
                pred_counts = torch.histc(pred_labels.float(),
                                          bins=self.num_classes, min=0, max=self.num_classes - 1)

                point_cloud_stat_true.append((true_counts.unsqueeze(0), len(voxel_points)))
                point_cloud_stat_pre.append((pred_counts.unsqueeze(0), len(voxel_points)))

        return mask, point_cloud_stat_true, point_cloud_stat_pre

class BADGESelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 13,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'mean'):
        self.select_num = select_num
        self.num_classes = num_classes
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.voxel_select_method = voxel_select_method

    def select(self, mask, raw_coord, preds, true_label, selected_class_count, features_before_c_1):
        true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=mask.device, dtype=torch.int)
        point_cloud_stat_true = []
        point_cloud_stat_pre = []
        voxel_record = None

        if self.select_method == 'voxel':
            with torch.no_grad():
                # Process predictions for 13 classes (assuming last channel is ignored)
                valid_preds = preds[:, :self.num_classes]
                pseudo_labels = valid_preds.argmax(dim=1)
                one_hot_pseudo = F.one_hot(pseudo_labels, self.num_classes).float()
                conf = F.softmax(valid_preds, dim=1)
                grad_logits = conf - one_hot_pseudo
                features = features_before_c_1.view(-1, features_before_c_1.shape[-1])
                D = features.shape[1]

                # Create gradient embeddings
                gradient_embeddings = (features.unsqueeze(2) * grad_logits.unsqueeze(1)).view(-1, D * self.num_classes)

            # Voxelization
            voxelized_coord, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels = point_in_voxel.shape[0]
            valid_mask = point_in_voxel != -1

            # Aggregate features per voxel
            voxel_embeddings = torch.zeros((num_voxels, D * self.num_classes), device=features.device)
            for voxel_id in range(num_voxels):
                point_ids = point_in_voxel[voxel_id][valid_mask[voxel_id]]
                if point_ids.numel() == 0:
                    continue
                if self.voxel_select_method == 'max':
                    agg_embed = gradient_embeddings[point_ids].max(dim=0).values
                else:
                    agg_embed = gradient_embeddings[point_ids].mean(dim=0)
                voxel_embeddings[voxel_id] = agg_embed

            # Get candidate voxels (not selected yet)
            expanded_mask = mask[point_in_voxel]  # (num_voxels, max_pts_per_voxel)
            expanded_mask[~valid_mask] = False
            candidate_mask = ~expanded_mask.any(dim=1)
            candidate_embeddings = voxel_embeddings[candidate_mask].cpu().numpy()

            if candidate_embeddings.shape[0] > 0:
                # K-means++ selection
                n_select = min(self.select_num, candidate_embeddings.shape[0])
                _, selected_indices = kmeans_plusplus(candidate_embeddings, n_select, random_state=42)
                selected_voxels = torch.where(candidate_mask)[0][selected_indices]

                # Process selected voxels
                voxel_record = []
                for voxel_id in selected_voxels:
                    # Get points in this voxel
                    point_ids = point_in_voxel[voxel_id][valid_mask[voxel_id]]
                    new_points = point_ids[~mask[point_ids]]

                    if new_points.numel() > 0:
                        mask[new_points] = True
                        voxel_record.append(voxelized_coord[voxel_id])

                        # Collect statistics
                        true_labels = true_label[new_points].clamp(0, self.num_classes)
                        pred_labels = valid_preds[new_points].argmax(dim=1).clamp(0, self.num_classes - 1)

                        # True counts
                        unique_true, counts_true = torch.unique(true_labels, return_counts=True)
                        true_counts = torch.zeros(self.num_classes + 1,
                                                  device=mask.device,
                                                  dtype=torch.int32)  # Explicit dtype
                        unique_true, counts_true = torch.unique(true_labels, return_counts=True)
                        true_counts[unique_true] = counts_true.to(torch.int32)  # Convert to match dtype

                        # Predicted label counts (0-12)
                        pred_counts = torch.zeros(self.num_classes + 1,
                                                  device=mask.device,
                                                  dtype=torch.int32)  # Explicit dtype
                        unique_pred, counts_pred = torch.unique(pred_labels, return_counts=True)
                        pred_counts[unique_pred] = counts_pred.to(torch.int32)  # Convert to match dtype

                        # Append stats
                        point_cloud_stat_true.append((true_counts.unsqueeze(0), len(new_points)))
                        point_cloud_stat_pre.append((pred_counts.unsqueeze(0), len(new_points)))
                        true_selected_label_count += true_counts.unsqueeze(0)

                voxel_record = torch.stack(voxel_record) if voxel_record else None

        return mask, point_cloud_stat_true, point_cloud_stat_pre, true_selected_label_count, voxel_record


class EntropySelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100, ):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            conf = F.softmax(preds, dim=-1)
            log2_conf = torch.log2(conf)
            entropy = -torch.mul(conf, log2_conf).sum(dim=1)
            entropy[-1] = 0

            point_entropy_in_voxel = entropy[point_in_voxel]
            # if self.voxel_select_method == 'max':
            voxel_entropy = torch.max(point_entropy_in_voxel, dim=1)[0]
            # elif self.voxel_select_method == 'mean':
            #     voxel_entropy = torch.sum(point_entropy_in_voxel, dim=1)
            #     voxel_entropy = voxel_entropy / voxel_point_counts
            # else:
            #     raise NotImplementedError

            _, voxel_indices = torch.sort(voxel_entropy, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            class_num_count_cpu = point_in_voxel_label_true.cpu()  # Move to CPU
            class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(0, self.num_classes + 1):
                # x = index[:self.select_num]
                class_num_count_i = torch.where(point_in_voxel_label_true[index[:self.select_num][0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                true_selected_label_count[:, i] = class_num_count_i
            return mask, true_selected_label_count
        else:
            raise NotImplementedError


class MarginSelect:
    def __init__(self,
                 select_num: int = 1,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.voxel_select_method = voxel_select_method

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            conf = F.softmax(preds, dim=-1)
            top2_conf, _ = torch.topk(conf, k=2, largest=True, dim=1, sorted=True)
            sub_result = top2_conf[:, 0] - top2_conf[:, 1]

            point_conf_in_voxel = sub_result[point_in_voxel]
            if self.voxel_select_method == 'max':
                voxel_conf = torch.max(point_conf_in_voxel, dim=1)[0]
            elif self.voxel_select_method == 'mean':
                voxel_conf = torch.sum(point_conf_in_voxel, dim=1)
                voxel_conf = voxel_conf / voxel_point_counts
            else:
                raise NotImplementedError

            _, voxel_indices = torch.sort(voxel_conf, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class VCDSelect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class VCDSelect_variance:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':

            # Initialize confusion score for each voxel across all batches
            batch_confusion = []

            for batch_index in range(preds.shape[1]):  # Iterating through each batch
                voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                    raw_coord, self.voxel_size, self.max_points_per_voxel)

                preds_label = preds[:, batch_index, :].max(dim=-1).indices
                point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)

                class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                              device=preds_label.device, dtype=torch.int)

                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=1)
                    class_num_count[:, i] = class_num_count_i

                class_num_probability = class_num_count / voxel_point_counts[:, None]
                temp = torch.log2(class_num_probability)
                log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                         torch.tensor(0, device=preds_label.device, dtype=torch.float))
                confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

                batch_confusion.append(confusion)

            # Average confusion across all batches
            average_confusion = torch.stack(batch_confusion).mean(dim=0)

            # Sort based on average confusion and update mask
            _, voxel_indices = torch.sort(average_confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask

        else:
            raise NotImplementedError


class Logit_Variance:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds
            # Create an output tensor initialized with -1.0 (float) to match the dtype of preds_label
            # class_num_count_cpu = preds_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.full(point_in_voxel.shape, -1.0, dtype=torch.float32).to(preds_label.device)

            # Create a mask where point_in_voxel is not equal to -1
            new_mask = (point_in_voxel != -1).to(preds_label.device)

            # Use the mask to index both point_in_voxel_label and preds_label
            # Assign the corresponding float values from preds_label to point_in_voxel_label
            point_in_voxel_label[new_mask] = preds_label[point_in_voxel[new_mask]]

            valid_mask = point_in_voxel_label != -1

            # Calculate the sum of each voxel using the mask
            voxel_sums = torch.where(valid_mask, point_in_voxel_label, torch.zeros_like(point_in_voxel_label)).sum(
                dim=1)

            # Count the valid entries in each voxel
            valid_counts = valid_mask.sum(dim=1)

            # Compute the average of each voxel, handling cases where the count is zero to avoid division by zero
            voxel_averages = voxel_sums / valid_counts.where(valid_counts != 0, torch.ones_like(valid_counts))

            # Handling NaN values if any voxel had all -1 (and thus a count of 0 leading to division by zero)
            voxel_averages[
                valid_counts == 0] = -1  # You can choose to set these to any specific value you deem appropriate

            _, voxel_indices = torch.sort(voxel_averages, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class VCDSelect_statistic:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            class_num_count_cpu = class_num_count.cpu()  # Move to CPU
            class_num_count_numpy = class_num_count_cpu.numpy()
            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_cloud_stat_true = []
            point_cloud_stat_pre = []
            pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                             dtype=torch.int)
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_true[:, i] = class_num_count_i
                class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_predicted[:, i] = class_num_count_i
            point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
            point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
            return mask, point_cloud_stat_true, point_cloud_stat_pre
        else:
            raise NotImplementedError


class Logit_Select_Reverse:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, stacked_out=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)

            # Continue as in your original code
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_cloud_stat_true = []
            point_cloud_stat_pre = []
            pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
            pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                             dtype=torch.int)
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_true[:, i] = class_num_count_i
                class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_predicted[:, i] = class_num_count_i
            point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
            point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
            return mask, point_cloud_stat_true, point_cloud_stat_pre
        else:
            raise NotImplementedError


class Logit_Select_Reverse_variance:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, stacked_out=None):
        if self.select_method == 'voxel':
            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Compute the max values and their corresponding indices for each voxel
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)

            # Squeeze the majority label indices
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels in stacked_out
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=stacked_out.device,
                                          dtype=torch.float16)

            # Loop through each possible class and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = stacked_out[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort voxels by mean logit in descending order
            _, voxel_indices = torch.sort(mean_logit, descending=True)

            # Select the indices for the first self.select_num voxels where the mask is False
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]

            # Update the mask for the selected voxels
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
        else:
            raise NotImplementedError


class Entropy_Greedy_Select:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try smallest entropy
                max_entropy_index = sorted_indices[0]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError


class Features_Entropy_Greedy_Select_Logit_Mean_point:
    def __init__(self,
                 select_num: int = 5,  # Set default to 5 points
                 num_classes: int = 20,
                 select_method: str = 'point',  # Change default to 'point'
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    # ... [Keep existing methods unchanged] ...

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'point':
            # Directly use point features without voxelization
            all_features = features_before_c_1.cpu().numpy()

            # Use selector to choose points
            selector = CustomSelection(n_samples=self.select_num, function=self.variance_score,
                                       optimizer='stochastic', verbose=True)
            selector.fit(all_features)
            selected_indices = selector.ranking[:self.select_num]

            # Convert to tensor and update mask
            selected_indices_tensor = torch.tensor(selected_indices, device=mask.device, dtype=torch.long)
            mask[selected_indices_tensor] = True
            mask[-1] = False  # Maintain last element as False

            # Get predicted and true labels for selected points
            preds_label = preds.max(dim=-1).indices
            selected_labels_pred = preds_label[selected_indices_tensor]
            selected_labels_true = true_label[selected_indices_tensor]

            # Count class occurrences
            pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1,
                                                        device=preds.device, dtype=torch.int)
            pre_selected_label_count_pred = torch.zeros(1, self.num_classes + 1,
                                                        device=preds.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                count_true = (selected_labels_true == i).sum()
                pre_selected_label_count_true[0, i] = count_true
                count_pred = (selected_labels_pred == i).sum()
                pre_selected_label_count_pred[0, i] = count_pred

            # Prepare statistics
            point_cloud_stat_true = [(pre_selected_label_count_true, self.select_num)]
            point_cloud_stat_pre = [(pre_selected_label_count_pred, self.select_num)]

            return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, None

        elif self.select_method == 'voxel':
            # ... [Keep existing voxel processing unchanged] ...
            pass
        else:
            raise NotImplementedError

class Features_Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)


    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Calculate the average, avoiding division by zero
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = F.normalize(feature_in_voxel_avg, p=2, dim=1)

            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            selector = CustomSelection(n_samples=200, function=self.variance_score, optimizer='stochastic',
                                       verbose=True)
            # selector = FacilityLocationSelection(n_samples=2000, metric='euclidean', optimizer='stochastic', verbose=True)
            # selector = MaxCoverageSelection(n_samples=2000, verbose=True)

            # Fit the selector to your data
            selector.fit(all_features)

            # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
            selected_indices = selector.ranking

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, voxelized_coordinates[index[max_entropy_index]]
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, None
        else:
            raise NotImplementedError

class Features_Entropy_Greedy_Select_Logit_Mean_majority:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)


    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Calculate the average, avoiding division by zero
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = F.normalize(feature_in_voxel_avg, p=2, dim=1)

            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            selector = CustomSelection(n_samples=2000, function=self.entropy_based_gradients, optimizer='stochastic',
                                       verbose=True)
            # selector = FacilityLocationSelection(n_samples=2000, metric='euclidean', optimizer='stochastic', verbose=True)
            # selector = MaxCoverageSelection(n_samples=2000, verbose=True)

            # Fit the selector to your data
            selector.fit(all_features)

            # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
            selected_indices = selector.ranking

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)
            new_label_matrix = point_in_voxel_label.clone()

            # Replace all non-(-1) labels with the corresponding majority label
            non_negative_indices = new_label_matrix != -1
            new_label_matrix[non_negative_indices] = majority_labels.unsqueeze(1).expand_as(new_label_matrix)[
                non_negative_indices]
            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(new_label_matrix[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, voxelized_coordinates[index[max_entropy_index]]
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, None
        else:
            raise NotImplementedError

class Features_Entropy_Greedy_Select_Logit_Mean_collect:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)

    def count_uniform_label_rows(self, tensor):
        # Initialize a counter to track the number of rows that meet the condition
        uniform_row_count = 0

        # Iterate over each row in the tensor
        for row in tensor:
            # Filter out `-1` values
            filtered_values = row[row != -1]

            # Check if the filtered array has exactly one unique element or is empty
            if filtered_values.unique().size(0) == 1 or filtered_values.nelement() == 0:
                # Increment the counter if the condition is met
                uniform_row_count += 1

        return uniform_row_count

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            count_uniform_voxel = self.count_uniform_label_rows(point_in_voxel_label_true)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]
            count_total_voxel = num_voxels
            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Calculate the average, avoiding division by zero
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = F.normalize(feature_in_voxel_avg, p=2, dim=1)

            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            selector = CustomSelection(n_samples=200, function=self.entropy_based_gradients, optimizer='stochastic',
                                       verbose=True)
            # selector = FacilityLocationSelection(n_samples=2000, metric='euclidean', optimizer='stochastic', verbose=True)
            # selector = MaxCoverageSelection(n_samples=2000, verbose=True)

            # Fit the selector to your data
            selector.fit(all_features)

            # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
            selected_indices = selector.ranking

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, count_uniform_voxel, count_total_voxel
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true, count_uniform_voxel, count_total_voxel
        else:
            raise NotImplementedError

class Features_Entropy_Greedy_Select_Logit_Mean_Kmeans:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)


    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Assuming `summed_features` and `non_zero_counts` are already defined and calculated
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = torch.nn.functional.normalize(feature_in_voxel_avg, p=2, dim=1)

            # Convert to NumPy array for compatibility with sklearn
            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            all_features = all_features[:2000]
            # Apply k-means clustering to select 200 samples
            kmeans = KMeans(n_clusters=200, random_state=42)
            kmeans.fit(all_features)

            # Select one sample from each cluster (the sample closest to each cluster center)
            selected_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(all_features - center, axis=1)
                selected_indices.append(np.argmin(distances))

            selected_indices = np.array(selected_indices)

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
        else:
            raise NotImplementedError

class Features_Entropy_Greedy_Select_Logit_Mean_Kmeans_plus_plus:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)


    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Assuming `summed_features` and `non_zero_counts` are already defined and calculated
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = torch.nn.functional.normalize(feature_in_voxel_avg, p=2, dim=1)

            # Convert to NumPy array for compatibility with sklearn
            all_features = feature_in_voxel_avg_normalized.cpu().numpy()

            # Apply k-means clustering with k-means++ initialization to select 200 samples
            kmeans = KMeans(n_clusters=200, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(all_features)

            # Select one sample from each cluster (the sample closest to each cluster center)
            selected_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(all_features - center, axis=1)
                selected_indices.append(np.argmin(distances))

            selected_indices = np.array(selected_indices)

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
        else:
            raise NotImplementedError

class Features_Entropy_Greedy_Select_Logit_Mean_gmm:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)


    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Assuming `summed_features` and `non_zero_counts` are already defined and calculated
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = torch.nn.functional.normalize(feature_in_voxel_avg, p=2, dim=1)

            # Convert to NumPy array for compatibility with sklearn
            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            all_features = all_features[:200]
            # Apply Gaussian Mixture Model to select 200 samples
            gmm = GaussianMixture(n_components=10, covariance_type='tied', random_state=42, init_params='kmeans')
            gmm.fit(all_features)

            # Predict the cluster for each sample and then select one sample per cluster
            cluster_labels = gmm.predict(all_features)

            selected_indices = []
            for cluster in range(10):
                # Find indices of all points assigned to the current cluster
                indices_in_cluster = np.where(cluster_labels == cluster)[0]
                # Calculate the density (probability) of each point in the cluster
                densities = gmm.score_samples(all_features[indices_in_cluster])
                # Select the point with the highest density
                selected_index = indices_in_cluster[np.argmax(densities)]
                selected_indices.append(selected_index)

            selected_indices = np.array(selected_indices)

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
        else:
            raise NotImplementedError

class Both_Majarity_Features_Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def norm_score(self, X, norm_type=2):
        """
        Compute the norm-based score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.
        norm_type (int): The type of norm to compute (default is L2 norm).

        Returns:
        float: The norm score.
        """
        norm = np.linalg.norm(X, ord=norm_type)
        return norm

    def variance_score(self, X):
        """
        Compute the variance score of a given matrix or vector.

        Parameters:
        X (numpy.ndarray): A 1D or 2D array representing the data.

        Returns:
        float: The variance score.
        """
        return np.var(X)


    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~(point_in_voxel != -1)] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and valid mask
            valid_mask = point_in_voxel != -1
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Calculate mask for majority features
            majority_mask = (point_in_voxel_label == majority_labels.unsqueeze(1))

            # Combine valid mask and majority mask to focus only on majority and valid points
            combined_mask = valid_mask & majority_mask

            # Initialize tensor for majority features, using majority and valid indices
            majority_features = torch.zeros_like(feature_in_voxel)
            majority_features[combined_mask] = features_before_c_1[clipped_indices[combined_mask].long()]

            # Create a mask for non-zero majority features
            non_zero_majority_mask = (majority_features != 0).any(dim=2).to(feature_in_voxel.device)

            # Sum the majority features across points, ignoring zero tensors
            summed_majority_features = (majority_features * non_zero_majority_mask.unsqueeze(2)).sum(dim=1)

            # Count the number of valid (non-zero) majority points in each voxel
            non_zero_majority_counts = non_zero_majority_mask.sum(dim=1).clamp(min=1)  # Clamp to avoid division by zero

            # Calculate the average of majority features, avoiding division by zero
            feature_in_voxel_majority_avg = summed_majority_features / non_zero_majority_counts.unsqueeze(1)

            # Normalize the averaged majority features
            feature_in_voxel_majority_avg_normalized = F.normalize(feature_in_voxel_majority_avg, p=2, dim=1)

            # Convert to numpy or any other format as required for further processing
            all_majority_features = feature_in_voxel_majority_avg_normalized.cpu().numpy()

            selector = CustomSelection(n_samples=2000, function=self.entropy_based_gradients, optimizer='stochastic',
                                       verbose=True)
            # selector = FacilityLocationSelection(n_samples=2000, metric='euclidean', optimizer='stochastic', verbose=True)
            # selector = MaxCoverageSelection(n_samples=2000, verbose=True)

            # Fit the selector to your data
            selector.fit(all_majority_features)

            # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
            selected_indices = selector.ranking

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            first_appear_indices = torch.full((self.num_classes + 1,), -1, device='cuda:0')

            # Loop through the sorted indices and populate the first_appear_indices tensor
            for idx in index:
                if first_appear_indices[-1] != -1:
                    break
                class_label = majority_labels[idx]
                if first_appear_indices[class_label] == -1:  # Check if it's the first appearance
                    first_appear_indices[class_label] = idx
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(len(first_appear_indices)):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    # for i in range(1, self.num_classes + 1):
                    #     class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                    #     class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    #     pre_selected_label_count[:, i] = class_num_count_i
                    current_label = majority_labels[first_appear_indices[ith]]
                    current_count = torch.sum(class_num_count[first_appear_indices[ith]])
                    pre_selected_label_count[:, current_label] = current_count
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try smallest entropy
                max_entropy_index = first_appear_indices[sorted_indices[-1]]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre, pre_selected_label_count_true
        else:
            raise NotImplementedError


class No_stage_1_Features_Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre
        else:
            raise NotImplementedError


class No_stage_3_Features_Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None,
               features_before_c_1=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Calculate the average, avoiding division by zero
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = F.normalize(feature_in_voxel_avg, p=2, dim=1)

            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            selector = CustomSelection(n_samples=2000, function=self.entropy_based_gradients, optimizer='stochastic',
                                       verbose=True)
            # selector = FacilityLocationSelection(n_samples=2000, metric='euclidean')

            # Fit the selector to your data
            selector.fit(all_features)

            # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
            selected_indices = selector.ranking

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]

            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            point_cloud_stat_true = []
            point_cloud_stat_pre = []
            pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                        dtype=torch.int)
            pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                             dtype=torch.int)
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_true[:, i] = class_num_count_i
                class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=0)
                pre_selected_label_count_predicted[:, i] = class_num_count_i
            point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
            point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
            return mask, point_cloud_stat_true, point_cloud_stat_pre
        else:
            raise NotImplementedError


class Second_inference_Features_Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def entropy_based_gradients(self, X):
        # Normalize gradients to probability distributions along features
        if np.sum(X) == 0:
            return 0  # Avoid division by zero
        normalized_gradients = X / np.sum(X, axis=0, keepdims=True)
        entropy = -np.sum(normalized_gradients * np.log(normalized_gradients + 1e-9), axis=0)
        return np.sum(entropy)

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None, model=None,
               batch=None, optimizer=None, features_before_c_1=None):
        voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
            raw_coord, self.voxel_size, self.max_points_per_voxel)
        # model.train()
        optimizer.zero_grad()
        ret_dict = model(batch, is_active=True, is_training=True)
        # voxel_label = batch['targets'].F.long().cuda(non_blocking=True)
        voxel_predict_logits_active = ret_dict['stacked_predict_logits']
        pseudo_voxel_label = voxel_predict_logits_active.max(dim=-1).indices
        # pseudo_voxel_label_current = pseudo_voxel_label[batch['targets'].C[:, 3] == 0]
        # pseudo_voxel_label_1 = pseudo_voxel_label_current[batch['inverse_map'][0]]
        network_loss = ret_dict['network_loss']
        # preds_label = pseudo_voxel_label_1
        # # class_num_count_cpu = preds.cpu()  # Move to CPU
        # # class_num_count_numpy = class_num_count_cpu.numpy()
        # point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel],
        #                                    -1)
        #
        # class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
        #                               device=preds_label.device, dtype=torch.int)
        # max_number_of_point_per_voxel = point_in_voxel_label.size(1)
        #
        # # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
        # # class_num_count_numpy = class_num_count_cpu.numpy()
        # for i in range(1, self.num_classes + 1):
        #     class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
        #     class_num_count_i = torch.sum(class_num_count_i, dim=1)
        #     class_num_count[:, i] = class_num_count_i
        # max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
        # # Gather the majority labels for each voxel
        # majority_labels = max_index.squeeze(1)
        #
        # # Find the maximum point index in point_in_voxel to define the size of point_labels
        # num_points = point_in_voxel.max().item() + 1  # Assuming points are indexed starting from 0
        #
        # # Initialize point_labels with a default value (e.g., -1) for all possible points
        # point_labels = torch.full((num_points,), -1, dtype=majority_labels.dtype,
        #                           device=majority_labels.device)
        #
        # # Flatten point_in_voxel to get a 1D list of all point indices and create a valid mask
        # flat_point_indices = point_in_voxel.view(-1)
        # valid_mask = flat_point_indices != -1
        #
        # # Flatten the majority_labels for each point to match flat_point_indices
        # voxel_labels_for_points = majority_labels.repeat_interleave(
        #     point_in_voxel.size(1)).view(-1)
        #
        # # Assign each point its corresponding voxel label, only where the mask is valid
        # point_labels[flat_point_indices[valid_mask]] = voxel_labels_for_points[valid_mask]
        #
        # # Initialize pseudo_voxel_label_current_1 with a default value
        # pseudo_voxel_label_current_1 = torch.full((len(voxel_label),), -1,
        #                                           dtype=point_labels.dtype,
        #                                           device=point_labels.device)
        #
        # # Convert inverse_map and pseudo_voxel_label_1 to tensors if they are not
        # inverse_map_tensor = torch.tensor(batch['inverse_map'][0],
        #                                   device=point_labels.device)
        # pseudo_voxel_label_1_tensor = torch.tensor(point_labels,
        #                                            device=point_labels.device)
        #
        # # Sort inverse_map_tensor and pseudo_voxel_label_1_tensor by inverse_map to control which occurrence is kept
        # # Sorting in descending order keeps the last occurrence; ascending keeps the first
        # sorted_indices = torch.argsort(inverse_map_tensor,
        #                                descending=True)  # Change to False for the first occurrence
        # sorted_inverse_map = inverse_map_tensor[sorted_indices]
        # sorted_labels = pseudo_voxel_label_1_tensor[sorted_indices]
        #
        # # Use scatter to assign values to pseudo_voxel_label_current_1
        # pseudo_voxel_label_current_1.scatter_(0, sorted_inverse_map, sorted_labels)
        # default_label = 0
        # pseudo_voxel_label_current_1[pseudo_voxel_label_current_1 == -1] = default_label
        initial_weights = copy.deepcopy(model.state_dict())
        loss = network_loss(ret_dict['voxel_predict_logits'], pseudo_voxel_label)
        # grad_test = torch.autograd.grad(loss, list(self.model.parameters()))
        optimizer.zero_grad()
        loss.backward()
        # # Step 3: Backpropagate with scaled loss and temporarily update weights
        # self.optimizer.zero_grad()  # Clear previous gradients
        # self.scaler.scale(loss).backward()  # Scaled backward pass for mixed precision
        # self.scaler.unscale_(self.optimizer)  # Unscale gradients for clipping
        # clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)  # Apply gradient clipping
        # self.scaler.step(self.optimizer)  # Temporarily update model weights
        # self.scaler.update()
        # self.scheduler.step()

        # Step 4: Perform inference to capture features with the temporarily updated model
        with torch.no_grad():  # Disable gradient tracking for inference
            ret_dict = model(batch, is_active=True, is_training=True)  # Run forward pass to get features
        features_before_c = ret_dict['features']
        features_before_c_current = features_before_c[batch['targets'].C[:, 3] == 0]
        features_before_c_1 = features_before_c_current[batch['inverse_map'][0]]
        # Step 5: Restore the initial weights to revert changes
        model.load_state_dict(initial_weights)
        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            num_voxels, points_per_voxel = point_in_voxel.shape
            feature_dim = features_before_c_1.shape[1]

            # Initialize an empty tensor for feature_in_voxel with zeros where points are invalid (-1)
            feature_in_voxel = torch.zeros((num_voxels, points_per_voxel, feature_dim), device=preds.device,
                                           dtype=torch.float16)

            # Mask to find valid indices (where index is not -1)
            valid_mask = point_in_voxel != -1

            # Replace -1 indices with 0 for indexing purposes, as they will be ignored by the mask
            clipped_indices = point_in_voxel.clone()
            clipped_indices[~valid_mask] = 0  # Set invalid indices to 0 to prevent out-of-bounds indexing

            # Gather features based on indices and mask
            feature_in_voxel[valid_mask] = features_before_c_1[clipped_indices[valid_mask].long()]

            # Create a mask for non-zero features (features not entirely zero) on the same device as feature_in_voxel
            non_zero_mask = (feature_in_voxel != 0).any(dim=2).to(feature_in_voxel.device)  # Shape: (29131, 12)

            # Sum the features across points, ignoring zero tensors
            summed_features = (feature_in_voxel * non_zero_mask.unsqueeze(2)).sum(dim=1)  # Shape: (29131, 96)

            # Count the number of valid (non-zero) points in each voxel
            non_zero_counts = non_zero_mask.sum(dim=1).clamp(min=1)  # Shape: (29131,); clamp to avoid division by zero

            # Calculate the average, avoiding division by zero
            feature_in_voxel_avg = summed_features / non_zero_counts.unsqueeze(1)
            feature_in_voxel_avg_normalized = F.normalize(feature_in_voxel_avg, p=2, dim=1)

            all_features = feature_in_voxel_avg_normalized.cpu().numpy()
            selector = CustomSelection(n_samples=2000, function=self.entropy_based_gradients, optimizer='stochastic',
                                       verbose=True)

            # Fit the selector to your data
            selector.fit(all_features)

            # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
            selected_indices = selector.ranking

            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=point_in_voxel.device)

            # Create the new point_in_voxel by indexing
            point_in_voxel = point_in_voxel[selected_indices_tensor]

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try max entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[sorted_indices[-1]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                point_cloud_stat_true = []
                point_cloud_stat_pre = []
                pre_selected_label_count_true = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                            dtype=torch.int)
                pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                                 dtype=torch.int)
                for i in range(1, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_true[:, i] = class_num_count_i
                    class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    pre_selected_label_count_predicted[:, i] = class_num_count_i
                point_cloud_stat_true.append((pre_selected_label_count_true, max_number_of_point_per_voxel))
                point_cloud_stat_pre.append((pre_selected_label_count_predicted, max_number_of_point_per_voxel))
                return mask, point_cloud_stat_true, point_cloud_stat_pre
        else:
            raise NotImplementedError


class Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try smallest entropy
                max_entropy_index = sorted_indices[0]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError


# entropy 用 majarity
class Majarity_Reverse_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]

            first_appear_indices = torch.full((self.num_classes + 1,), -1, device='cuda:0')

            # Loop through the sorted indices and populate the first_appear_indices tensor
            for idx in index:
                if first_appear_indices[-1] != -1:
                    break
                class_label = majority_labels[idx]
                if first_appear_indices[class_label] == -1:  # Check if it's the first appearance
                    first_appear_indices[class_label] = idx
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(len(first_appear_indices)):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    # for i in range(1, self.num_classes + 1):
                    #     class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                    #     class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    #     pre_selected_label_count[:, i] = class_num_count_i
                    current_label = majority_labels[first_appear_indices[ith]]
                    current_count = torch.sum(class_num_count[first_appear_indices[ith]])
                    pre_selected_label_count[:, current_label] = current_count
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try smallest entropy
                max_entropy_index = first_appear_indices[sorted_indices[-1]]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError


class VCD_Entropy_Greedy_Select_Logit_Mean:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            # class_num_count_cpu = preds.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)
            max_number_of_point_per_voxel = point_in_voxel_label.size(1)

            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()
            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)
            class_num_count_cpu = confusion.cpu()  # Move to CPU
            class_num_count_numpy = class_num_count_cpu.numpy()
            max_values, max_index = torch.max(class_num_count, dim=1, keepdim=True)
            # Gather the majority labels for each voxel
            majority_labels = max_index.squeeze(1)

            # Create a mask to filter out invalid points (-1)
            valid_points_mask = point_in_voxel != -1

            # Create a tensor to hold logits for the majority labels
            majority_logits = torch.zeros(point_in_voxel.shape[0], point_in_voxel.shape[1], device=preds.device,
                                          dtype=torch.float16)

            # Loop through each voxel and gather logits for the majority label for each valid point in that voxel
            for i in range(self.num_classes + 1):
                mask_for_label = (majority_labels == i)[:,
                                 None] & valid_points_mask  # Mask for voxels with majority label 'i'
                selected_voxels = torch.masked_select(point_in_voxel, mask_for_label).view(
                    -1)  # Indices of points in voxels
                if len(selected_voxels) > 0:
                    logits_for_label = preds[selected_voxels, i]  # Gather logits for the selected points
                    majority_logits[mask_for_label] = logits_for_label

            # Compute mean logit for each voxel by averaging over valid points
            valid_point_counts = valid_points_mask.sum(dim=1).clamp(min=1)  # Prevent division by zero
            mean_logit = majority_logits.sum(dim=1) / valid_point_counts

            # Sort by mean logit
            _, voxel_indices = torch.sort(mean_logit, descending=False)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            index = index[:50]
            selected_confusion_values = confusion[index]

            # Sort these selected confusion values in descending order
            sorted_values, sorted_indices = torch.sort(selected_confusion_values, descending=True)

            # Retrieve the corresponding original indices using the sorted indices
            sorted_original_indices = index[sorted_indices]

            index = sorted_original_indices[:10]
            pre_selected_label_entropy = []
            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(10):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try smallest entropy
                max_entropy_index = sorted_indices[0]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError


class Location_Entropy_Max:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            # class_num_count_cpu = current_entropy.cpu()  # Move to CPU
            # class_num_count_numpy_2 = class_num_count_cpu.numpy()
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_cluster_tratio = []

            if not selected_class_count:

                for ith in range(10):
                    coordinate_list = []
                    label_list = []
                    current_index = voxel_indices[ith]
                    for _, tensor in enumerate(point_in_voxel[current_index]):
                        if tensor == -1:
                            break
                        coordinate_list.append((raw_coord[tensor].cpu()))
                        label_list.append((preds_label[tensor].cpu()))
                    cluster_tratio = self.compute_cluster_ratios(coordinate_list, label_list)
                    pre_selected_label_cluster_tratio.append(cluster_tratio)

                sorted_indices = sorted(range(len(pre_selected_label_cluster_tratio)),
                                        key=lambda i: pre_selected_label_cluster_tratio[i])
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError

    def compute_cluster_ratios(self, points, labels):
        """
        Computes the mean of the cluster ratios, where each cluster ratio is defined as the
        average distance from all points in the cluster to the cluster's centroid divided
        by the diagonal of the bounding box enclosing all points.

        Parameters:
            points (torch.Tensor): Tensor of shape (n, 3) where n is the number of points and 3 are their coordinates.
            labels (torch.Tensor): Tensor of shape (n,) where n is the number of points, containing cluster labels.

        Returns:
            torch.Tensor: The mean of cluster ratios.
        """
        points = torch.stack(points)
        labels = torch.tensor(labels)
        unique_classes = labels.unique()
        centroids = {cls.item(): torch.mean(points[labels == cls], dim=0) for cls in unique_classes}
        cluster_ratios = []

        # Compute the bounding box diagonal as a normalization factor
        min_point = torch.min(points, dim=0).values
        max_point = torch.max(points, dim=0).values
        bounding_box_diagonal = torch.norm(max_point - min_point)

        for cls in unique_classes:
            class_points = points[labels == cls]
            class_centroid = centroids[cls.item()]
            distances = torch.norm(class_points - class_centroid, dim=1)
            average_distance = torch.mean(distances)
            cluster_ratio = average_distance / bounding_box_diagonal
            cluster_ratios.append(cluster_ratio)

        return torch.mean(torch.stack(cluster_ratios))


class Location_Entropy_Overlap:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            # class_num_count_cpu = current_entropy.cpu()  # Move to CPU
            # class_num_count_numpy_2 = class_num_count_cpu.numpy()
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_cluster_tratio = []

            if not selected_class_count:

                for ith in range(10):
                    coordinate_list = []
                    label_list = []
                    current_index = voxel_indices[ith]
                    for _, tensor in enumerate(point_in_voxel[current_index]):
                        if tensor == -1:
                            break
                        coordinate_list.append((raw_coord[tensor].cpu()))
                        label_list.append((preds_label[tensor].cpu()))
                    cluster_tratio = self.compute_overlap_ratios(coordinate_list, label_list)
                    pre_selected_label_cluster_tratio.append(cluster_tratio)

                sorted_indices = sorted(range(len(pre_selected_label_cluster_tratio)),
                                        key=lambda i: pre_selected_label_cluster_tratio[i])
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError

    def compute_overlap_ratios(self, points, labels):
        """
        Computes the mean of the cluster ratios, where each cluster ratio is defined as the
        average distance from all points in the cluster to the cluster's centroid divided
        by the diagonal of the bounding box enclosing all points.

        Parameters:
            points (torch.Tensor): Tensor of shape (n, 3) where n is the number of points and 3 are their coordinates.
            labels (torch.Tensor): Tensor of shape (n,) where n is the number of points, containing cluster labels.

        Returns:
            torch.Tensor: The mean of cluster ratios.
        """
        points = torch.stack(points)
        labels = torch.tensor(labels)
        unique_classes, counts = labels.unique(return_counts=True)
        centroids = {cls.item(): torch.mean(points[labels == cls], dim=0) for cls in unique_classes}
        cluster_ratios = []
        overlap_ratios = []
        class_centroid_list = []
        classes_with_multiple_instances = unique_classes[counts > 1]

        # Compute the bounding box diagonal as a normalization factor
        min_point = torch.min(points, dim=0).values
        max_point = torch.max(points, dim=0).values
        bounding_box_diagonal = torch.norm(max_point - min_point)

        for cls in unique_classes:
            class_points = points[labels == cls]
            class_centroid = centroids[cls.item()]
            class_centroid_list.append(class_centroid.numpy())
            distances = torch.norm(class_points - class_centroid, dim=1)
            average_distance = torch.mean(distances)
            cluster_ratio = average_distance / bounding_box_diagonal
            cluster_ratios.append(cluster_ratio.numpy())

        centroids_array = np.stack(class_centroid_list, axis=0)
        cluster_ratios_array = np.stack(cluster_ratios, axis=0)
        est_vol = self.monte_carlo_volume(centroids_array, cluster_ratios_array)
        if est_vol != 0:
            print("hello")
        return est_vol  # torch.mean(torch.stack(cluster_ratios))

    def monte_carlo_volume(self, centers, radii, trials=2024):
        # Calculate bounds based on sphere centers and radii
        min_bounds = np.min(centers - radii[:, np.newaxis], axis=0)
        max_bounds = np.max(centers + radii[:, np.newaxis], axis=0)

        # Generate random points within the bounding box
        points = np.random.uniform(min_bounds, max_bounds, (trials, 3))

        # Function to check if a point is inside all spheres
        def is_inside_all_spheres(points):
            distances = np.sqrt(np.sum((points[:, np.newaxis, :] - centers) ** 2, axis=2))
            inside_all = np.all(distances <= radii, axis=1)
            return np.sum(inside_all)

        # Count how many points fall inside all of the spheres
        count_inside = is_inside_all_spheres(points)

        # Calculate the volume of the bounding box
        volume_of_bounds = np.prod(max_bounds - min_bounds)

        # Estimate the volume covered by the intersecting part of the spheres
        return volume_of_bounds * count_inside / trials


class Location_Entropy_Max_Entropy_Balancing:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            # class_num_count_cpu = point_in_voxel_label_true.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()

            class_num_count = torch.zeros((point_in_voxel_label_true.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label_true == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            # class_num_count_cpu = current_entropy.cpu()  # Move to CPU
            # class_num_count_numpy_2 = class_num_count_cpu.numpy()
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_cluster_tratio = []
            pre_selected_label_entropy = []
            if selected_class_count:

                for ith in range(20):
                    coordinate_list = []
                    label_list = []
                    current_index = voxel_indices[ith]
                    for _, tensor in enumerate(point_in_voxel[current_index]):
                        if tensor == -1:
                            break
                        coordinate_list.append((raw_coord[tensor].cpu()))
                        label_list.append((true_label[tensor].cpu()))
                    cluster_tratio = self.compute_cluster_ratios(coordinate_list, label_list)
                    pre_selected_label_cluster_tratio.append(cluster_tratio)

                sorted_indices = sorted(range(len(pre_selected_label_cluster_tratio)),
                                        key=lambda i: pre_selected_label_cluster_tratio[i])
                location_entropy_index = sorted_indices[-10:]
                selected_class_count = selected_class_count[0]
                for ith in range(len(location_entropy_index)):
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device,
                                                           dtype=torch.int)
                    for i in range(1, self.num_classes + 1):
                        class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[ith]] == i, 1, 0)
                        class_num_count_i = torch.sum(class_num_count_i, dim=0)
                        pre_selected_label_count[:, i] = class_num_count_i
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                max_entropy_index = sorted_indices[-1]

                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError

    def compute_cluster_ratios(self, points, labels):
        """
        Computes the mean of the cluster ratios, where each cluster ratio is defined as the
        average distance from all points in the cluster to the cluster's centroid divided
        by the diagonal of the bounding box enclosing all points.

        Parameters:
            points (torch.Tensor): Tensor of shape (n, 3) where n is the number of points and 3 are their coordinates.
            labels (torch.Tensor): Tensor of shape (n,) where n is the number of points, containing cluster labels.

        Returns:
            torch.Tensor: The mean of cluster ratios.
        """
        points = torch.stack(points)
        labels = torch.tensor(labels)
        unique_classes = labels.unique()
        centroids = {cls.item(): torch.mean(points[labels == cls], dim=0) for cls in unique_classes}
        cluster_ratios = []

        # Compute the bounding box diagonal as a normalization factor
        min_point = torch.min(points, dim=0).values
        max_point = torch.max(points, dim=0).values
        bounding_box_diagonal = torch.norm(max_point - min_point)

        for cls in unique_classes:
            class_points = points[labels == cls]
            class_centroid = centroids[cls.item()]
            distances = torch.norm(class_points - class_centroid, dim=1)
            average_distance = torch.mean(distances)
            cluster_ratio = average_distance / bounding_box_diagonal
            cluster_ratios.append(cluster_ratio)

        return torch.mean(torch.stack(cluster_ratios))


class Location_Entropy_Max_Centroid_distance:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':
            true_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            # class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            # class_num_count_numpy = class_num_count_cpu.numpy()

            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            # class_num_count_cpu = current_entropy.cpu()  # Move to CPU
            # class_num_count_numpy_2 = class_num_count_cpu.numpy()
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_cluster_tratio = []

            if not selected_class_count:

                for ith in range(10):
                    coordinate_list = []
                    label_list = []
                    current_index = voxel_indices[ith]
                    for _, tensor in enumerate(point_in_voxel[current_index]):
                        if tensor == -1:
                            break
                        coordinate_list.append((raw_coord[tensor].cpu()))
                        label_list.append((preds_label[tensor].cpu()))
                    cluster_tratio = self.compute_cluster_ratios(coordinate_list, label_list)
                    pre_selected_label_cluster_tratio.append(cluster_tratio)

                sorted_indices = sorted(range(len(pre_selected_label_cluster_tratio)),
                                        key=lambda i: pre_selected_label_cluster_tratio[i])
                max_entropy_index = sorted_indices[0]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
        else:
            raise NotImplementedError

    def compute_cluster_ratios(self, points, labels):
        """
        Computes the mean of the cluster ratios, where each cluster ratio is defined as the
        average distance from all points in the cluster to the cluster's centroid divided
        by the diagonal of the bounding box enclosing all points.

        Parameters:
            points (torch.Tensor): Tensor of shape (n, 3) where n is the number of points and 3 are their coordinates.
            labels (torch.Tensor): Tensor of shape (n,) where n is the number of points, containing cluster labels.

        Returns:
            torch.Tensor: The mean of cluster ratios.
        """
        points = torch.stack(points)
        labels = torch.tensor(labels)
        unique_classes = labels.unique()
        centroids = {cls.item(): torch.mean(points[labels == cls], dim=0) for cls in unique_classes}
        cluster_ratios = []

        # Compute the bounding box diagonal as a normalization factor
        min_point = torch.min(points, dim=0).values
        max_point = torch.max(points, dim=0).values
        bounding_box_diagonal = torch.norm(max_point - min_point)

        for cls in unique_classes:
            class_points = points[labels == cls]
            class_centroid = centroids[cls.item()]
            distances = torch.norm(class_points - class_centroid, dim=1)
            average_distance = torch.mean(distances)
            cluster_ratio = average_distance / bounding_box_diagonal
            cluster_ratios.append(cluster_ratio)

        return torch.mean(torch.stack(cluster_ratios))


class Density_Entropy_Greedy_Select_Overall:
    def __init__(self,
                 select_num: int = 1,
                 num_classes: int = 20,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'max'):
        self.select_num = select_num
        self.select_method = select_method
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_classes = num_classes

    def select(self, mask, raw_coord=None, preds=None, true_label=None, selected_class_count=None):

        if self.select_method == 'voxel':

            voxelized_coordinates, voxel_idx, inverse, point_in_voxel, voxel_point_counts = get_point_in_voxel(
                raw_coord, self.voxel_size, self.max_points_per_voxel)

            preds_label = preds.max(dim=-1).indices
            point_in_voxel_label = torch.where(point_in_voxel != -1, preds_label[point_in_voxel], -1)
            true_selected_label_count = torch.zeros(1, 100, device=preds.device, dtype=torch.int)
            point_in_voxel_label_true = torch.where(point_in_voxel != -1, true_label[point_in_voxel], -1)
            class_num_count_cpu = point_in_voxel_label.cpu()  # Move to CPU
            class_num_count_numpy = class_num_count_cpu.numpy()
            class_num_count = torch.zeros((point_in_voxel_label.shape[0], self.num_classes + 1),
                                          device=preds_label.device, dtype=torch.int)

            for i in range(1, self.num_classes + 1):
                class_num_count_i = torch.where(point_in_voxel_label == i, 1, 0)
                class_num_count_i = torch.sum(class_num_count_i, dim=1)
                class_num_count[:, i] = class_num_count_i

            class_num_probability = class_num_count / voxel_point_counts[:, None]
            temp = torch.log2(class_num_probability)
            log2_class_num_probability = torch.where(class_num_count != 0, temp,
                                                     torch.tensor(0, device=preds_label.device, dtype=torch.float))
            confusion = -torch.mul(class_num_probability, log2_class_num_probability).sum(dim=1)

            _, voxel_indices = torch.sort(confusion, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            pre_selected_label_entropy = []

            if selected_class_count:
                selected_class_count = selected_class_count[0]
                for ith in range(5):
                    pre_selected_label_count = torch.zeros(1, 100, device=preds.device, dtype=torch.int)
                    if point_in_voxel_label_true[voxel_indices[0]][point_in_voxel_label.shape[1] - 1] != -1:
                        pre_selected_label_count[0, point_in_voxel_label.shape[1] - 1] += 1
                    else:
                        for i in range(point_in_voxel_label.shape[1]):
                            if point_in_voxel_label_true[voxel_indices[0]][i] == -1:
                                pre_selected_label_count[0, i - 1] += 1
                    pre_selected_label_count += selected_class_count
                    # Calculate probabilities from counts
                    probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()

                    # Remove zero probabilities to avoid log(0) which is undefined
                    probabilities = probabilities[probabilities > 0]

                    # Calculate entropy
                    entropy = -torch.sum(probabilities * torch.log2(probabilities))
                    pre_selected_label_entropy.append(entropy)

                sorted_indices = sorted(range(len(pre_selected_label_entropy)),
                                        key=lambda i: pre_selected_label_entropy[i])
                # try biggest entropy
                max_entropy_index = sorted_indices[-1]
                mask[point_in_voxel[index[max_entropy_index]]] = True
                mask[-1] = False
                for i in range(0, self.num_classes + 1):
                    class_num_count_i = torch.where(point_in_voxel_label_true[voxel_indices[0]] == i, 1, 0)
                    class_num_count_i = torch.sum(class_num_count_i, dim=0)
                    true_selected_label_count[:, i] = class_num_count_i
                return mask, true_selected_label_count
            else:
                mask[point_in_voxel[index[:self.select_num]]] = True
                mask[-1] = False
                if point_in_voxel_label_true[voxel_indices[0]][point_in_voxel_label.shape[1] - 1] != -1:
                    true_selected_label_count[0, point_in_voxel_label.shape[1] - 1] += 1
                else:
                    for i in range(point_in_voxel_label.shape[1]):
                        if point_in_voxel_label_true[voxel_indices[0]][i] == -1:
                            true_selected_label_count[0, i - 1] += 1  # Increment only the i-th element

                return mask, true_selected_label_count
        else:
            raise NotImplementedError
