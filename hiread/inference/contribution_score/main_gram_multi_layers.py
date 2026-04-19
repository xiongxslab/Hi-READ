#!/usr/bin/env python3
"""
多layer GRAM分析脚本
支持同时分析多个层的GRAM贡献
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
# matplotlib imports removed - using visualization.py for professional plotting

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import model_utils

# 尝试导入可视化模块，如果失败则跳过
try:
    from hiread.inference.contribution_score.visualization import ComprehensiveVisualizer
    HAS_VISUALIZER = True
except ImportError:
    print("警告: 无法导入ComprehensiveVisualizer，将跳过专业可视化功能")
    HAS_VISUALIZER = False

# 旧的create_multi_layer_gram_visualization函数已删除，现在使用visualization.py中的专业函数

class MultiLayerGRAMAnalyzer:
    """多layer GRAM分析器"""
    
    def __init__(self, model, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.activation_maps = {}
        self.gradients = {}
        self.hooks = []
        
        # 注册hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """为所有目标层注册hooks"""
        for layer_name in self.target_layers:
            target_layer = self._get_target_layer(layer_name)
            if target_layer is not None:
                print(f"已注册hooks到层: {layer_name}")
                
                # 激活hook
                def get_activation_hook(layer_name):
                    def hook(module, input, output):
                        self.activation_maps[layer_name] = output
                    return hook
                
                # 梯度hook
                def get_gradient_hook(layer_name):
                    def hook(module, grad_input, grad_output):
                        self.gradients[layer_name] = grad_output[0]
                    return hook
                
                hook_act = target_layer.register_forward_hook(get_activation_hook(layer_name))
                hook_grad = target_layer.register_full_backward_hook(get_gradient_hook(layer_name))
                self.hooks.extend([hook_act, hook_grad])
            else:
                print(f"警告: 未找到层 {layer_name}")
    
    def _get_target_layer(self, layer_name: str):
        """获取目标层"""
        print(f"查找目标层: {layer_name}")
        print("可用层:")
        for name, module in self.model.named_modules():
            if 'gate' in name.lower() or 'fusion' in name.lower() or 'encoder' in name.lower() or 'conv' in name.lower():
                print(f"  {name}: {type(module)}")
            if name == layer_name:
                print(f"找到目标层: {name}")
                return module
        
        # 如果没找到，尝试查找包含关键词的层
        for name, module in self.model.named_modules():
            if layer_name in name:
                print(f"使用替代层: {name}")
                return module
        
        print(f"未找到目标层: {layer_name}")
        return None
    
    def compute_gram_for_all_layers(self, input_tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """为所有目标层计算GRAM"""
        print(f"\n=== 开始多layer GRAM计算 ===")
        print(f"目标层: {self.target_layers}")
        print(f"输入张量形状: {input_tensor.shape}")
        
        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)
        print(f"模型输出形状: {output.shape}")
        
        # 反向传播
        output.sum().backward()
        
        results = {}
        
        for layer_name in self.target_layers:
            if layer_name in self.activation_maps and layer_name in self.gradients:
                print(f"\n--- 计算 {layer_name} 的GRAM ---")
                
                activation_maps = self.activation_maps[layer_name]
                gradients = self.gradients[layer_name]
                
                print(f"激活图形状: {activation_maps.shape}")
                print(f"梯度形状: {gradients.shape}")
                
                # 计算Alpha权重
                alpha_weights = self._compute_alpha_weights(activation_maps, gradients)
                print(f"Alpha权重形状: {alpha_weights.shape}")
                print(f"Alpha权重范围: [{alpha_weights.min():.4f}, {alpha_weights.max():.4f}]")
                
                # 计算GRAM
                gram_1d = self._compute_gram_1d(activation_maps, alpha_weights)
                
                results[layer_name] = (gram_1d, alpha_weights)
                
                print(f"{layer_name} GRAM完成。形状: {gram_1d.shape}")
            else:
                print(f"警告: {layer_name} 的激活图或梯度未找到")
        
        return results
    
    def _compute_alpha_weights(self, activation_maps: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """计算Alpha权重"""
        batch_size, num_channels, length = activation_maps.shape
        Z = length
        
        alpha_weights = torch.zeros(num_channels, device=activation_maps.device)
        
        for k in range(num_channels):
            alpha_k = gradients[:, k, :].sum(dim=1) / Z
            alpha_weights[k] = alpha_k.mean()
        
        return alpha_weights
    
    def _compute_gram_1d(self, activation_maps: torch.Tensor, alpha_weights: torch.Tensor) -> torch.Tensor:
        """计算1D GRAM"""
        batch_size, num_channels, length = activation_maps.shape
        
        print(f"计算GRAM: {num_channels}个通道, {length}个位置")
        
        # 1. ReLU(alpha_weights)
        alpha_relu = F.relu(alpha_weights)  # [channels]
        
        # 2. ReLU(activations)
        activations_relu = F.relu(activation_maps)  # [batch, channels, length]
        
        # 3. 正确的GRAM计算: GRAM_m(r) = Σ_k ReLU(α_k^r) · ReLU(A_k^m)
        gram_1d = torch.zeros(batch_size, length, device=activation_maps.device)
        
        for k in range(num_channels):
            alpha_k = alpha_relu[k]
            activation_k = activations_relu[:, k, :]  # [batch, length]
            gram_1d += alpha_k * activation_k
        
        print(f"GRAM 计算完成。形状: {gram_1d.shape}")
        return gram_1d.squeeze(0)  # 移除batch维度
    
    def cleanup(self):
        """清理hooks"""
        for hook in self.hooks:
            hook.remove()

class ContributionAnalyzer:
    """贡献分析器主类"""
    
    def __init__(self, chr_name: str, start_pos: int, model_path: str, 
                 seq_path: str, chip_path: str, celltype: str, output_path: str,
                 perturb_width: int = 8192, step_size: int = 8192):
        
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.model_path = model_path
        self.seq_path = seq_path
        self.chip_path = chip_path
        self.celltype = celltype
        self.output_path = output_path
        self.perturb_width = perturb_width
        self.step_size = step_size
        
        # 创建目录
        self.save_path = f'{output_path}/{celltype}/contribution_analysis'
        for subdir in ['imgs', 'npy', 'attention', 'gradcam', 'impact_scores', 'multi_layer_gram']:
            os.makedirs(f'{self.save_path}/{subdir}', exist_ok=True)
        
        # 初始化
        self._load_model_and_data()
        
    def _load_model_and_data(self):
        """加载模型和数据"""
        print(f"加载模型: {self.model_path}")
        self.model = model_utils.load_default(self.model_path)
        self.model.eval()
        
        print(f"加载数据: {self.chr_name}:{self.start_pos}")
        # 构建完整的序列文件路径
        seq_file_path = os.path.join(self.seq_path, f'{self.chr_name}.fa.gz')
        print(f"序列文件路径: {seq_file_path}")
        self.seq_region, self.chip_region = infer.load_region(
            self.chr_name, self.start_pos, seq_file_path, self.chip_path)
        
        print(f"序列形状: {self.seq_region.shape}, ChIP-seq形状: {self.chip_region.shape}")
        
    def compute_prediction(self) -> np.ndarray:
        """计算Hi-C预测"""
        print("计算Hi-C预测...")
        inputs = infer.preprocess_default(self.seq_region, self.chip_region)
        
        with torch.no_grad():
            pred = self.model(inputs)[0].detach().cpu().numpy()
        
        np.save(f'{self.save_path}/npy/prediction.npy', pred)
        return pred
    
    def compute_impact_scores(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """计算impact scores"""
        print("计算impact scores...")
        
        try:
            screen_start = self.start_pos
            screen_end = self.start_pos + 2097152
            
            print(f"Screen范围: {screen_start}-{screen_end}")
            print(f"Perturb width: {self.perturb_width}, Step: {self.step_size}")
            
            # 构建完整的序列文件路径
            seq_file_path = os.path.join(self.seq_path, f'{self.chr_name}.fa.gz')
            print(f"Impact score序列文件路径: {seq_file_path}")
            
            seq, chip = infer.load_data_default(self.chr_name, seq_file_path, self.chip_path)
            model = model_utils.load_default(self.model_path)
            
            windows = [w * self.step_size + screen_start 
                      for w in range(int((screen_end - screen_start) / self.step_size))]
            
            impact_scores = []
            perturb_starts = []
            perturb_ends = []
            
            print(f"计算{len(windows)}个窗口...")
            
            for w_start in windows:
                pred_start = int(w_start + self.perturb_width / 2 - 2097152 / 2)
                
                try:
                    from hiread.inference import screening
                    pred, pred_deletion, diff_map = screening.predict_difference(
                        self.chr_name, pred_start, int(w_start), 
                        self.perturb_width, model, seq, chip)
                    
                    impact_score = np.abs(diff_map.mean())
                    impact_scores.append(impact_score)
                    perturb_starts.append(w_start)
                    perturb_ends.append(w_start + self.perturb_width)
                    
                except Exception as e:
                    print(f"窗口{w_start}失败: {e}")
                    continue
            
            impact_scores = np.array(impact_scores)
            perturb_starts = np.array(perturb_starts)
            perturb_ends = np.array(perturb_ends)
            
            # 保存结果
            np.save(f'{self.save_path}/impact_scores/impact_scores.npy', impact_scores)
            np.save(f'{self.save_path}/impact_scores/perturb_starts.npy', perturb_starts)
            np.save(f'{self.save_path}/impact_scores/perturb_ends.npy', perturb_ends)
            
            print(f"Impact scores完成: {len(impact_scores)}个点")
            
            return impact_scores, perturb_starts, perturb_ends
            
        except Exception as e:
            print(f"Impact score计算失败: {e}")
            return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Multi-layer GRAM Analysis')
    
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
    parser.add_argument('--layers', dest='target_layers', nargs='+', 
                        default=['encoder.gate_fusion', 'encoder.encoder_seq', 'encoder.encoder_epi'],
                        help='GRAM分析的目标层列表')
    
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-layer GRAM Analysis")
    print("=" * 60)
    print(f"分析区域: {args.chr_name}:{args.start}")
    print(f"细胞类型: {args.celltype}")
    print(f"模型路径: {args.model_path}")
    print(f"目标层: {args.target_layers}")
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
        
        # 4. 多layer GRAM分析
        gram_results = {}
        if not args.skip_gram:
            print("\n3. 计算多layer GRAM分析...")
            # 准备输入数据
            inputs = infer.preprocess_default(analyzer.seq_region, analyzer.chip_region)
            
            # 创建多layer GRAM分析器
            gram_analyzer = MultiLayerGRAMAnalyzer(analyzer.model, args.target_layers)
            
            try:
                # 计算所有层的GRAM
                gram_results = gram_analyzer.compute_gram_for_all_layers(inputs)
                
                # 保存和转换结果
                for layer_name, (gram_1d, alpha_weights) in gram_results.items():
                    # 保存原始结果
                    np.save(f'{analyzer.save_path}/multi_layer_gram/{layer_name}_gram_1d.npy', 
                           gram_1d.detach().cpu().numpy())
                    np.save(f'{analyzer.save_path}/multi_layer_gram/{layer_name}_alpha_weights.npy', 
                           alpha_weights.detach().cpu().numpy())
                    
                    # 转换为numpy数组用于可视化
                    gram_results[layer_name] = (
                        gram_1d.detach().cpu().numpy(),
                        alpha_weights.detach().cpu().numpy()
                    )
                    
                    print(f"{layer_name} GRAM分析完成。GRAM形状: {gram_1d.shape}")
                    print(f"{layer_name} GRAM值范围: [{gram_1d.min():.4f}, {gram_1d.max():.4f}]")
                
            finally:
                # 清理hooks
                gram_analyzer.cleanup()
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
        
        # 创建imgs目录
        imgs_dir = f'{analyzer.save_path}/imgs'
        os.makedirs(imgs_dir, exist_ok=True)
        
        # 使用专业的可视化工具
        if gram_results and not args.skip_gram and HAS_VISUALIZER:
            print("   创建多层GRAM专业可视化...")
            
            # 创建可视化器
            visualizer = ComprehensiveVisualizer(
                chr_name=args.chr_name,
                start_pos=args.start,
                save_path=analyzer.save_path
            )
            
            # 转换gram_results格式
            gram_data = {}
            for layer_name, (gram_1d, alpha_weights) in gram_results.items():
                if hasattr(gram_1d, 'detach'):
                    gram_data[layer_name] = gram_1d.detach().cpu().numpy()
                else:
                    gram_data[layer_name] = gram_1d
            
            # 创建多layer GRAM的paper style可视化
            visualizer.create_paper_style_visualization_multi_gram(
                prediction=prediction,
                chipseq_data=chipseq_data,
                gram_results=gram_data,
                impact_scores=impact_scores,
                perturb_starts=perturb_starts,
                step_size=args.step_size
            )
            
            # 创建多layer比较可视化
            visualizer.create_multi_layer_gram_comparison(
                prediction=prediction,
                chipseq_data=chipseq_data,
                gram_results=gram_data,
                impact_scores=impact_scores,
                perturb_starts=perturb_starts,
                step_size=args.step_size
            )
            
            # 创建单独的track可视化
            visualizer.create_individual_tracks_gram(
                prediction=prediction,
                chipseq_data=chipseq_data,
                gram_1d=None,  # 单个GRAM，这里不需要
                alpha_weights=None,
                impact_scores=impact_scores,
                perturb_starts=perturb_starts
            )

        
        # 7. 生成分析报告
        print("\n6. 生成分析报告...")
        generate_analysis_report_multi_layer(
            analyzer.save_path, args, prediction, chipseq_data, 
            gram_results, impact_scores
        )
        
        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"所有结果保存在: {analyzer.save_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

def generate_analysis_report_multi_layer(save_path: str, args, prediction: np.ndarray, 
                                       chipseq_data: np.ndarray, 
                                       gram_results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                       impact_scores: Optional[np.ndarray]):
    """生成多layer分析报告"""
    
    report_path = f'{save_path}/analysis_report_multi_layer_gram.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Multi-layer GRAM Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"分析区域: {args.chr_name}:{args.start}\n")
        f.write(f"细胞类型: {args.celltype}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"目标层: {args.target_layers}\n")
        f.write(f"扰动参数: width={args.perturb_width}bp, step={args.step_size}bp\n\n")
        
        # Hi-C预测统计
        f.write("Hi-C预测统计:\n")
        f.write(f"  预测矩阵形状: {prediction.shape}\n")
        f.write(f"  预测值范围: [{prediction.min():.4f}, {prediction.max():.4f}]\n")
        f.write(f"  预测均值: {prediction.mean():.4f}\n")
        f.write(f"  预测标准差: {prediction.std():.4f}\n\n")
        
        # ChIP-seq数据统计
        f.write("ChIP-seq数据统计:\n")
        f.write(f"  数据形状: {chipseq_data.shape}\n")
        f.write(f"  信号范围: [{chipseq_data.min():.4f}, {chipseq_data.max():.4f}]\n")
        f.write(f"  信号均值: {chipseq_data.mean():.4f}\n")
        f.write(f"  信号标准差: {chipseq_data.std():.4f}\n\n")
        
        # 多layer GRAM分析结果
        f.write("多layer GRAM分析结果:\n")
        f.write("=" * 40 + "\n")
        
        for layer_name, (gram_1d, alpha_weights) in gram_results.items():
            f.write(f"\n{layer_name}:\n")
            f.write(f"  GRAM形状: {gram_1d.shape}\n")
            f.write(f"  GRAM值范围: [{gram_1d.min():.4f}, {gram_1d.max():.4f}]\n")
            f.write(f"  GRAM均值: {gram_1d.mean():.4f}\n")
            f.write(f"  GRAM标准差: {gram_1d.std():.4f}\n")
            f.write(f"  Alpha权重形状: {alpha_weights.shape}\n")
            f.write(f"  Alpha权重范围: [{alpha_weights.min():.4f}, {alpha_weights.max():.4f}]\n")
            f.write(f"  Alpha权重均值: {alpha_weights.mean():.4f}\n")
            f.write(f"  Alpha权重标准差: {alpha_weights.std():.4f}\n")
            
            # 计算GRAM峰值位置
            peak_positions = np.where(gram_1d > np.percentile(gram_1d, 90))[0]
            f.write(f"  GRAM峰值位置 (top 10%): {len(peak_positions)}个位置\n")
            if len(peak_positions) > 0:
                f.write(f"  主要峰值位置: {peak_positions[:10].tolist()}\n")
        
        # Impact scores统计
        if impact_scores is not None:
            f.write(f"\nImpact Scores统计:\n")
            f.write(f"  Impact scores数量: {len(impact_scores)}\n")
            f.write(f"  Impact scores范围: [{impact_scores.min():.4f}, {impact_scores.max():.4f}]\n")
            f.write(f"  Impact scores均值: {impact_scores.mean():.4f}\n")
            f.write(f"  Impact scores标准差: {impact_scores.std():.4f}\n\n")
        
        # 层间比较分析
        f.write("层间比较分析:\n")
        f.write("=" * 40 + "\n")
        
        if len(gram_results) > 1:
            gram_means = {}
            gram_stds = {}
            gram_maxs = {}
            
            for layer_name, (gram_1d, _) in gram_results.items():
                gram_means[layer_name] = gram_1d.mean()
                gram_stds[layer_name] = gram_1d.std()
                gram_maxs[layer_name] = gram_1d.max()
            
            f.write("GRAM统计比较:\n")
            f.write("  层名称\t\t均值\t\t标准差\t\t最大值\n")
            for layer_name in gram_results.keys():
                f.write(f"  {layer_name:<20}\t{gram_means[layer_name]:.4f}\t\t{gram_stds[layer_name]:.4f}\t\t{gram_maxs[layer_name]:.4f}\n")
            
            # 找出贡献最大的层
            max_mean_layer = max(gram_means.items(), key=lambda x: x[1])
            max_std_layer = max(gram_stds.items(), key=lambda x: x[1])
            max_max_layer = max(gram_maxs.items(), key=lambda x: x[1])
            
            f.write(f"\n分析结论:\n")
            f.write(f"  平均贡献最大的层: {max_mean_layer[0]} (均值: {max_mean_layer[1]:.4f})\n")
            f.write(f"  变化最大的层: {max_std_layer[0]} (标准差: {max_std_layer[1]:.4f})\n")
            f.write(f"  峰值最大的层: {max_max_layer[0]} (最大值: {max_max_layer[1]:.4f})\n")
        
        # 总结
        f.write(f"\n总结:\n")
        f.write("=" * 40 + "\n")
        f.write(f"本次分析成功计算了 {len(gram_results)} 个层的GRAM贡献。\n")
        f.write(f"每个层都提供了不同角度的模型解释:\n")
        
        layer_descriptions = {
            'encoder.gate_fusion': '融合特征层，包含序列和表观遗传信息的融合',
            'encoder.encoder_seq': '序列编码器，专门分析DNA序列特征的贡献',
            'encoder.encoder_epi': '表观编码器，专门分析表观遗传特征的贡献',
            'encoder.conv_end': '最终卷积层，分析最终编码器输出的重要性'
        }
        
        for layer_name in gram_results.keys():
            desc = layer_descriptions.get(layer_name, '未知层')
            f.write(f"  - {layer_name}: {desc}\n")
        
        f.write(f"\n所有结果文件保存在: {save_path}\n")
        f.write(f"详细的可视化图像保存在: {save_path}/imgs/\n")
        f.write(f"原始数据保存在: {save_path}/multi_layer_gram/\n")
    
    print(f"详细分析报告保存到: {report_path}")

if __name__ == "__main__":
    main() 
