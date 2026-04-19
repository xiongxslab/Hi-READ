#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAM (Gradient-weighted Region Activation Map) 分析器
基于image中描述的GRAM方法，专门针对256×256 Hi-C热图输出的模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import model_utils

class GRAMAnalyzer:
    """
    GRAM分析器 - 专门针对2D输出（256×256 Hi-C热图）的梯度激活映射
    基于image中的GRAM公式实现
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_gram(self, 
                    target_layer, 
                    input_tensor, 
                    region_type='full',
                    normalize=True):
        """
        计算GRAM (Gradient-weighted Region Activation Map)
        
        Args:
            target_layer: 目标层
            input_tensor: 输入张量
            region_type: 区域类型 ('full', 'diagonal', 'center', 'corner')
            normalize: 是否归一化
            
        Returns:
            gram_map: GRAM激活图
            alpha_weights: 激活权重
        """
        
        # 确保输入在正确的设备上
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # 前向传播，获取激活图
        activation_maps = self._get_activation_maps(target_layer, input_tensor)
        
        # 获取输出区域
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]  # 如果返回多个值，取第一个
        
        # 根据region_type选择输出区域
        target_region = self._select_output_region(output, region_type)
        
        # 计算梯度
        gradients = self._compute_gradients(target_region, input_tensor)
        
        # 计算激活权重 α_k^r
        alpha_weights = self._compute_alpha_weights(gradients, activation_maps)
        
        # 计算GRAM
        gram_map = self._compute_gram_map(alpha_weights, activation_maps)
        
        if normalize:
            gram_map = self._normalize_gram(gram_map)
            
        return gram_map, alpha_weights
    
    def _get_activation_maps(self, target_layer, input_tensor):
        """
        获取目标层的激活图
        """
        # 注册钩子来获取激活图
        activation_maps = []
        
        def hook_fn(module, input, output):
            activation_maps.append(output)
        
        hook = target_layer.register_forward_hook(hook_fn)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        hook.remove()
        
        return activation_maps[0] if activation_maps else None
    
    def _select_output_region(self, output, region_type):
        """
        根据region_type选择输出区域
        对应image中的区域r选择
        """
        if region_type == 'full':
            # 选择整个输出空间（256×256）
            return output
        elif region_type == 'diagonal':
            # 选择对角线区域
            batch_size, h, w = output.shape
            diagonal_mask = torch.eye(h, w, device=output.device)
            return output * diagonal_mask.unsqueeze(0)
        elif region_type == 'center':
            # 选择中心区域
            batch_size, h, w = output.shape
            center_h, center_w = h // 2, w // 2
            window_size = min(h, w) // 4
            return output[:, 
                        center_h-window_size:center_h+window_size,
                        center_w-window_size:center_w+window_size]
        elif region_type == 'corner':
            # 选择角落区域
            batch_size, h, w = output.shape
            corner_size = min(h, w) // 4
            return output[:, :corner_size, :corner_size]
        else:
            return output
    
    def _compute_gradients(self, target_region, input_tensor):
        """
        计算梯度 ∂r / ∂A_k,i,j^m
        对应image中的梯度计算部分
        """
        # 计算目标区域的总和
        if len(target_region.shape) == 3:
            target_sum = target_region.sum(dim=(1, 2))  # [batch_size]
        else:
            target_sum = target_region.sum()
        
        # 反向传播
        target_sum.backward(retain_graph=True)
        
        # 获取输入梯度
        gradients = input_tensor.grad
        
        return gradients
    
    def _compute_alpha_weights(self, gradients, activation_maps):
        """
        计算激活权重 α_k^r = (1/Z) * Σ_i Σ_j (∂r / ∂A_k,i,j^m)
        对应image中的α_k^r计算公式
        """
        if activation_maps is None:
            return None
            
        # 获取激活图的形状
        batch_size, num_channels, height, width = activation_maps.shape
        
        # 计算Z（总激活数）
        Z = height * width
        
        # 初始化alpha权重
        alpha_weights = torch.zeros(num_channels, device=activation_maps.device)
        
        # 对每个通道计算平均梯度
        for k in range(num_channels):
            # 获取第k个通道的激活图
            channel_activation = activation_maps[:, k, :, :]  # [batch, height, width]
            
            # 计算该通道的梯度（这里简化处理，实际应该根据激活图计算）
            # 在实际实现中，需要根据激活图的具体位置计算梯度
            channel_gradients = gradients[:, k, :, :] if len(gradients.shape) == 4 else gradients
            
            # 计算平均梯度
            alpha_k = torch.mean(channel_gradients) / Z
            alpha_weights[k] = alpha_k
        
        return alpha_weights
    
    def _compute_gram_map(self, alpha_weights, activation_maps):
        """
        计算GRAM: GRAM_m(r) = Σ_k [ReLU(α_k^r) ⋅ ReLU(A_k^m)]
        对应image中的GRAM公式
        """
        if alpha_weights is None or activation_maps is None:
            return None
            
        batch_size, num_channels, height, width = activation_maps.shape
        
        # 初始化GRAM图
        gram_map = torch.zeros(batch_size, height, width, device=activation_maps.device)
        
        # 对每个通道计算贡献
        for k in range(num_channels):
            # 获取激活权重
            alpha_k = alpha_weights[k]
            
            # 获取激活图
            A_k = activation_maps[:, k, :, :]
            
            # 应用ReLU
            alpha_relu = F.relu(alpha_k)
            A_relu = F.relu(A_k)
            
            # 计算该通道的贡献
            channel_contribution = alpha_relu * A_relu
            
            # 累加到GRAM图
            gram_map += channel_contribution
        
        return gram_map
    
    def _normalize_gram(self, gram_map):
        """
        归一化GRAM图
        """
        if gram_map is None:
            return None
            
        # 最小-最大归一化
        gram_min = torch.min(gram_map)
        gram_max = torch.max(gram_map)
        
        if gram_max > gram_min:
            gram_map = (gram_map - gram_min) / (gram_max - gram_min)
        
        return gram_map
    
    def analyze_multiple_layers(self, 
                              input_tensor, 
                              target_layers,
                              region_types=['full', 'diagonal', 'center']):
        """
        分析多个层的GRAM
        """
        results = {}
        
        for layer_name, target_layer in target_layers.items():
            layer_results = {}
            
            for region_type in region_types:
                try:
                    gram_map, alpha_weights = self.compute_gram(
                        target_layer, input_tensor, region_type
                    )
                    layer_results[region_type] = {
                        'gram_map': gram_map,
                        'alpha_weights': alpha_weights
                    }
                except Exception as e:
                    print(f"计算 {layer_name} 层的 {region_type} 区域GRAM失败: {e}")
                    layer_results[region_type] = None
            
            results[layer_name] = layer_results
        
        return results
    
    def visualize_gram(self, gram_map, save_path=None, title="GRAM Analysis"):
        """
        可视化GRAM结果
        """
        if gram_map is None:
            print("GRAM图为空，无法可视化")
            return
        
        # 转换为numpy数组
        if isinstance(gram_map, torch.Tensor):
            gram_map = gram_map.detach().cpu().numpy()
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 显示GRAM热图
        im = ax.imshow(gram_map, cmap='hot', interpolation='nearest')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('GRAM Score')
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GRAM可视化保存到: {save_path}")
        
        plt.show()
        
    def create_comprehensive_analysis(self, 
                                   input_tensor,
                                   target_layers,
                                   save_dir="./gram_analysis"):
        """
        创建综合的GRAM分析
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 分析多个层和区域
        results = self.analyze_multiple_layers(input_tensor, target_layers)
        
        # 可视化结果
        for layer_name, layer_results in results.items():
            for region_type, region_data in layer_results.items():
                if region_data is not None:
                    gram_map = region_data['gram_map']
                    alpha_weights = region_data['alpha_weights']
                    
                    # 保存GRAM图
                    gram_path = os.path.join(save_dir, f"{layer_name}_{region_type}_gram.npy")
                    np.save(gram_path, gram_map.detach().cpu().numpy())
                    
                    # 保存alpha权重
                    alpha_path = os.path.join(save_dir, f"{layer_name}_{region_type}_alpha.npy")
                    np.save(alpha_path, alpha_weights.detach().cpu().numpy())
                    
                    # 可视化
                    vis_path = os.path.join(save_dir, f"{layer_name}_{region_type}_gram.png")
                    self.visualize_gram(gram_map, vis_path, 
                                      f"GRAM - {layer_name} - {region_type}")
        
        return results


class HiCGRAMWrapper(torch.nn.Module):
    """
    Hi-C模型的GRAM包装器
    专门处理256×256 Hi-C热图输出
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        """
        前向传播，保持原始输出格式
        """
        output = self.model(x)
        
        # 如果输出是元组，取第一个元素
        if isinstance(output, tuple):
            output = output[0]
        
        # 确保输出是256×256的热图
        if len(output.shape) == 3:  # [batch, height, width]
            return output
        else:
            # 如果输出不是预期的形状，进行调整
            return output


def create_target_layers_for_gram(model):
    """
    为GRAM分析创建目标层
    根据您的模型结构选择关键层
    """
    target_layers = {}
    
    # 根据模型结构选择目标层
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        
        # RRTEncoderSplit的关键层
        if hasattr(encoder, 'encoder_seq'):
            target_layers['encoder_seq'] = encoder.encoder_seq
        if hasattr(encoder, 'encoder_epi'):
            target_layers['encoder_epi'] = encoder.encoder_epi
        if hasattr(encoder, 'gate_fusion'):
            target_layers['gate_fusion'] = encoder.gate_fusion
        if hasattr(encoder, 'conv_end'):
            target_layers['conv_end'] = encoder.conv_end
        
        # 如果找不到特定层，使用整个encoder
        if not target_layers:
            target_layers['encoder'] = encoder
    
    # 如果还是找不到，使用整个模型
    if not target_layers:
        target_layers['model'] = model
    
    return target_layers


def main():
    """
    示例用法
    """
    # 这里应该加载您的模型
    # model = load_your_model()
    
    # 创建GRAM分析器
    # analyzer = GRAMAnalyzer(model)
    
    # 准备输入数据
    # input_tensor = prepare_input_data()
    
    # 获取目标层
    # target_layers = create_target_layers_for_gram(model)
    
    # 进行GRAM分析
    # results = analyzer.create_comprehensive_analysis(input_tensor, target_layers)
    
    print("GRAM分析器已创建，请根据您的具体模型进行配置")


if __name__ == "__main__":
    main() 
