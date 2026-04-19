#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Contribution Score Analysis主脚本 (GRAM版本)
整合Hi-C预测、ChIP-seq可视化、GRAM分析和Impact Score计算
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.contribution_score.main_gram_multi_layers import ContributionAnalyzer
from hiread.inference.contribution_score.visualization import ComprehensiveVisualizer
from hiread.inference.utils import inference_utils as infer

class GRAMAnalyzer:
    """GRAM分析器"""
    
    def __init__(self, model, target_layer_name="gate_fusion"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activation_maps = None
        self.gradients = None
        self.gram_1d = None
        
        # 注册hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向和反向hooks"""
        def get_activation_hook(module, input, output):
            self.activation_maps = output.detach()
            
        def get_gradient_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 获取目标层
        target_layer = self._get_target_layer()
        if target_layer is not None:
            target_layer.register_forward_hook(get_activation_hook)
            target_layer.register_backward_hook(get_gradient_hook)
            print(f"已注册hooks到层: {self.target_layer_name}")
        else:
            print(f"警告: 未找到目标层 {self.target_layer_name}")
    
    def _get_target_layer(self):
        """获取目标层"""
        print(f"查找目标层: {self.target_layer_name}")
        print("可用层:")
        for name, module in self.model.named_modules():
            if 'gate' in name.lower() or 'fusion' in name.lower():
                print(f"  {name}: {type(module)}")
            if name == self.target_layer_name:
                print(f"找到目标层: {name}")
                return module
        
        # 如果没找到，尝试查找包含关键词的层
        for name, module in self.model.named_modules():
            if 'gate_fusion' in name or 'gate' in name and 'fusion' in name:
                print(f"使用替代层: {name}")
                return module
        
        print(f"未找到目标层: {self.target_layer_name}")
        return None
    
    def compute_gram(self, input_tensor):
        """计算GRAM"""
        print(f"\n=== 开始GRAM计算 ===")
        print(f"输入张量形状: {input_tensor.shape}")
        
        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)
        print(f"模型输出形状: {output.shape}")
        
        # 获取目标层激活图
        target_layer = self._get_target_layer()
        if target_layer is None:
            raise ValueError(f"未找到目标层: {self.target_layer_name}")
        
        print(f"目标层激活图形状: {self.activation_maps.shape}")
        
        # 反向传播
        output.sum().backward()
        print(f"捕获梯度: {self.gradients.shape}")
        
        # 计算Alpha权重
        alpha_weights = self._compute_alpha_weights()
        print(f"Alpha权重形状: {alpha_weights.shape}")
        print(f"Alpha权重范围: [{alpha_weights.min():.4f}, {alpha_weights.max():.4f}]")
        
        # 计算GRAM
        gram_1d = self._compute_gram_1d(alpha_weights)
        
        return gram_1d, alpha_weights
    
    def _compute_alpha_weights(self):
        """计算Alpha权重"""
        batch_size, num_channels, length = self.activation_maps.shape
        Z = length
        
        alpha_weights = torch.zeros(num_channels, device=self.activation_maps.device)
        
        for k in range(num_channels):
            alpha_k = self.gradients[:, k, :].sum(dim=1) / Z
            alpha_weights[k] = alpha_k.mean()
        
        return alpha_weights
    
    def _compute_gram_1d(self, alpha_weights):
        """计算1D GRAM"""
        batch_size, num_channels, length = self.activation_maps.shape
        
        print(f"计算GRAM: {num_channels}个通道, {length}个位置")
        
        # 1. ReLU(alpha_weights)
        alpha_relu = F.relu(alpha_weights)  # [channels]
        
        # 2. ReLU(activations)
        activations_relu = F.relu(self.activation_maps)  # [batch, channels, length]
        
        # 3. 正确的GRAM计算: GRAM_m(r) = Σ_k ReLU(α_k^r) · ReLU(A_k^m)
        gram_1d = torch.zeros(batch_size, length, device=self.activation_maps.device)
        
        for k in range(num_channels):
            alpha_k = alpha_relu[k]
            activation_k = activations_relu[:, k, :]  # [batch, length]
            gram_1d += alpha_k * activation_k
        
        print(f"GRAM 计算完成。形状: {gram_1d.shape}")
        return gram_1d.squeeze(0)  # 移除batch维度

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Contribution Score Analysis with GRAM')
    
    # 基本参数
    parser.add_argument('--chr', dest='chr_name', required=True,
                        help='染色体名称 (例如: chr1)')
    parser.add_argument('--start', dest='start', type=int, required=True,
                        help='起始位置 (bp)')
    parser.add_argument('--model', dest='model_path', required=True,
                        help='模型文件路径')
    parser.add_argument('--seq', dest='seq_path', required=True,
                        help='序列文件夹路径')
    parser.add_argument('--chip', dest='chip_path', required=True,
                        help='ChIP-seq文件路径')
    parser.add_argument('--celltype', dest='celltype', required=True,
                        help='细胞类型')
    parser.add_argument('--out', dest='output_path', default='outputs',
                        help='输出路径')
    parser.add_argument('--target-layer', dest='target_layer', default='gate_fusion',
                        help='GRAM分析的目标层 (默认: gate_fusion)')
    
    # Impact score参数
    parser.add_argument('--perturb-width', dest='perturb_width', type=int, default=8192,
                        help='扰动宽度 (bp), 默认: 8192')
    parser.add_argument('--step-size', dest='step_size', type=int, default=8192,
                        help='步长 (bp), 默认: 8192')
    
    # 功能开关
    parser.add_argument('--skip-gram', dest='skip_gram', action='store_true',
                        help='跳过GRAM分析')
    parser.add_argument('--skip-impact', dest='skip_impact', action='store_true',
                        help='跳过Impact Score分析')
    parser.add_argument('--paper-style-only', dest='paper_style_only', action='store_true',
                        help='只生成论文样式的综合图')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comprehensive Contribution Score Analysis with GRAM")
    print("=" * 60)
    print(f"分析区域: {args.chr_name}:{args.start}")
    print(f"细胞类型: {args.celltype}")
    print(f"模型路径: {args.model_path}")
    print(f"目标层: {args.target_layer}")
    print(f"扰动参数: width={args.perturb_width}bp, step={args.step_size}bp")
    print("=" * 60)
    
    try:
        # 1. 创建分析器
        print("初始化分析器...")
        analyzer = ContributionAnalyzer(
            chr_name=args.chr_name,
            start_pos=args.start,
            model_path=args.model_path,
            seq_path=args.seq_path,
            chip_path=args.chip_path,
            celltype=args.celltype,
            output_path=args.output_path,
            perturb_width=args.perturb_width,
            step_size=args.step_size
        )
        
        # 2. 基础Hi-C预测
        print("\n1. 计算Hi-C预测...")
        prediction = analyzer.compute_prediction()
        
        # 3. ChIP-seq数据
        print("\n2. 准备ChIP-seq数据...")
        chipseq_data = analyzer.chip_region.copy()
        np.save(f'{analyzer.save_path}/npy/chipseq_track.npy', chipseq_data)
        
        # 4. GRAM分析
        gram_1d = None
        alpha_weights = None
        if not args.skip_gram:
            print("\n3. 计算GRAM分析...")
            # 准备输入数据
            inputs = infer.preprocess_default(analyzer.seq_region, analyzer.chip_region)
            gram_analyzer = GRAMAnalyzer(analyzer.model, args.target_layer)
            gram_1d, alpha_weights = gram_analyzer.compute_gram(inputs)
            
            # 保存GRAM结果
            np.save(f'{analyzer.save_path}/npy/gram_1d.npy', gram_1d.detach().cpu().numpy())
            np.save(f'{analyzer.save_path}/npy/alpha_weights.npy', alpha_weights.detach().cpu().numpy())
            
            # 转换为numpy数组用于可视化
            gram_1d = gram_1d.detach().cpu().numpy()
            alpha_weights = alpha_weights.detach().cpu().numpy()
            
            print(f"GRAM分析完成。GRAM形状: {gram_1d.shape}")
            print(f"GRAM值范围: [{gram_1d.min():.4f}, {gram_1d.max():.4f}]")
        else:
            print("\n3. 跳过GRAM分析")
        
        # 5. Impact scores
        impact_scores, perturb_starts, perturb_ends = None, None, None
        if not args.skip_impact:
            print("\n4. 计算Impact Scores...")
            impact_scores, perturb_starts, perturb_ends = analyzer.compute_impact_scores()
        else:
            print("\n4. 跳过Impact Score分析")
        
        # 6. 创建可视化
        print("\n5. 创建可视化...")
        
        visualizer = ComprehensiveVisualizer(
            chr_name=args.chr_name,
            start_pos=args.start,
            save_path=analyzer.save_path
        )
        
        # 论文样式综合图
        print("   创建论文样式综合图...")
        visualizer.create_paper_style_visualization_gram(
            prediction=prediction,
            chipseq_data=chipseq_data,
            gram_1d=gram_1d,
            impact_scores=impact_scores,
            perturb_starts=perturb_starts,
            step_size=args.step_size
        )
        
        # 单独track图（除非只要论文样式）
        if not args.paper_style_only:
            print("   创建单独track图...")
            visualizer.create_individual_tracks_gram(
                prediction=prediction,
                chipseq_data=chipseq_data,
                gram_1d=gram_1d,
                alpha_weights=alpha_weights,
                impact_scores=impact_scores,
                perturb_starts=perturb_starts
            )
        
        # 7. 生成分析报告
        print("\n6. 生成分析报告...")
        generate_analysis_report_gram(
            analyzer.save_path, args, prediction, chipseq_data,
            gram_1d, alpha_weights, impact_scores
        )
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"所有结果保存在: {analyzer.save_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_analysis_report_gram(save_path: str, args, prediction: np.ndarray, 
                                 chipseq_data: np.ndarray, gram_1d: Optional[np.ndarray],
                                 alpha_weights: Optional[np.ndarray], impact_scores: Optional[np.ndarray]):
    """生成详细的GRAM分析报告"""
    
    report_lines = [
        "Comprehensive Contribution Score Analysis Report (GRAM Version)",
        "=" * 60,
        "",
        "分析参数:",
        f"  区域: {args.chr_name}:{args.start}-{args.start+2097152}",
        f"  细胞类型: {args.celltype}",
        f"  模型: {args.model_path}",
        f"  序列路径: {args.seq_path}",
        f"  ChIP-seq路径: {args.chip_path}",
        f"  目标层: {args.target_layer}",
        f"  扰动宽度: {args.perturb_width} bp",
        f"  步长: {args.step_size} bp",
        "",
        "分析结果:",
        f"  Hi-C预测矩阵: {prediction.shape}",
        f"  ChIP-seq track: {chipseq_data.shape}",
        f"  GRAM分析: {'✓ 成功' if gram_1d is not None else '✗ 失败/跳过'}",
        f"  Impact scores: {'✓ 成功' if impact_scores is not None else '✗ 失败/跳过'}",
        "",
        "GRAM分析详情:",
    ]
    
    if gram_1d is not None:
        report_lines.extend([
            f"  GRAM track形状: {gram_1d.shape}",
            f"  GRAM值范围: {gram_1d.min():.4f} - {gram_1d.max():.4f}",
            f"  GRAM平均值: {gram_1d.mean():.4f}",
            f"  GRAM标准差: {gram_1d.std():.4f}",
        ])
    
    if alpha_weights is not None:
        report_lines.extend([
            f"  Alpha权重形状: {alpha_weights.shape}",
            f"  Alpha权重范围: {alpha_weights.min():.4f} - {alpha_weights.max():.4f}",
            f"  Alpha权重平均值: {alpha_weights.mean():.4f}",
            f"  Alpha权重标准差: {alpha_weights.std():.4f}",
        ])
    
    report_lines.extend([
        "",
        "输出文件:",
        "  图像文件:",
        f"    - 论文样式综合图: imgs/{args.chr_name}_{args.start}_paper_style_gram.png",
        f"    - 单独Hi-C图: imgs/hic_prediction_individual.png",
        f"    - ChIP-seq track图: imgs/chip_track_individual.png",
    ])
    
    if gram_1d is not None:
        report_lines.append("    - GRAM track图: imgs/gram_track_individual.png")
    
    if impact_scores is not None:
        report_lines.append("    - Impact scores图: imgs/impact_scores_individual.png")
    
    report_lines.extend([
        "",
        "  数据文件:",
        "    - Hi-C预测: npy/prediction.npy",
        "    - ChIP-seq数据: npy/chipseq_track.npy",
    ])
    
    if gram_1d is not None:
        report_lines.extend([
            "    - GRAM数据: npy/gram_1d.npy",
            "    - Alpha权重: npy/alpha_weights.npy",
        ])
    
    if impact_scores is not None:
        report_lines.extend([
            "    - Impact scores: impact_scores/impact_scores.npy",
            "    - 扰动起始位置: impact_scores/perturb_starts.npy",
            "    - 扰动结束位置: impact_scores/perturb_ends.npy",
        ])
    
    report_lines.extend([
        "",
        "分析统计:",
        f"  预测值范围: {prediction.min():.3f} - {prediction.max():.3f}",
        f"  CHIP信号范围: {chipseq_data.min():.3f} - {chipseq_data.max():.3f}",
    ])
    
    if impact_scores is not None and len(impact_scores) > 0:
        report_lines.extend([
            f"  Impact score数量: {len(impact_scores)}",
            f"  Impact score范围: {impact_scores.min():.3f} - {impact_scores.max():.3f}",
            f"  平均Impact score: {impact_scores.mean():.3f}",
        ])
    elif impact_scores is not None:
        report_lines.append("  Impact score数量: 0 (计算失败)")
    else:
        report_lines.append("  Impact scores: 未计算或跳过")
    
    # 保存报告
    with open(f'{save_path}/analysis_report_gram.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"详细分析报告保存到: {save_path}/analysis_report_gram.txt")

if __name__ == '__main__':
    main() 
