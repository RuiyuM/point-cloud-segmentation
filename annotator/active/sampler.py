import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import torch_scatter
import torchsparse
import torchsparse.nn.functional

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


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
            perm = torch.randperm(voxel_idx.shape[0])
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


class EntropySelect:
    def __init__(self,
                 select_num: int = 1,
                 select_method: str = 'voxel',
                 voxel_size: float = 0.25,
                 max_points_per_voxel: int = 100,
                 voxel_select_method: str = 'mean'):
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
            log2_conf = torch.log2(conf)
            entropy = -torch.mul(conf, log2_conf).sum(dim=1)
            entropy[-1] = 0

            point_entropy_in_voxel = entropy[point_in_voxel]
            if self.voxel_select_method == 'max':
                voxel_entropy = torch.max(point_entropy_in_voxel, dim=1)[0]
            elif self.voxel_select_method == 'mean':
                voxel_entropy = torch.sum(point_entropy_in_voxel, dim=1)
                voxel_entropy = voxel_entropy / voxel_point_counts
            else:
                raise NotImplementedError

            _, voxel_indices = torch.sort(voxel_entropy, descending=True)
            index = voxel_indices[mask[point_in_voxel[voxel_indices][:, 0]] == False]
            mask[point_in_voxel[index[:self.select_num]]] = True
            mask[-1] = False
            return mask
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
            point_in_voxel_label = torch.full(point_in_voxel.shape, -1.0, dtype=torch.float16).to(preds_label.device)

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
            pre_selected_label_count_predicted = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
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
                    pre_selected_label_count = torch.zeros(1, self.num_classes + 1, device=preds.device, dtype=torch.int)
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
        return est_vol #torch.mean(torch.stack(cluster_ratios))

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
                            true_selected_label_count[0, i-1] += 1  # Increment only the i-th element

                return mask, true_selected_label_count
        else:
            raise NotImplementedError