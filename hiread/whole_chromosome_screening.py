#!/usr/bin/env python3
"""
全染色体GRAM和Impact Score扫描脚本
两阶段分析：
1. 8KB bin级别扫描识别候选区域
2. 候选区域内1KB精细扫描定位调控元件
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import model_utils
from hiread.inference import screening
from hiread.inference.contribution_score.main_gram_multi_layers import MultiLayerGRAMAnalyzer

class WholeChromosomeAnalyzer:
    """全染色体分析器"""
    
    def __init__(self, chr_name: str, model_path: str, seq_path: str, 
                 chip_path: str, celltype: str, output_path: str,
                 window_size: int = 2097152,  # 2MB窗口 = 256 bins
                 step_size: int = 1048576,    # 1MB步长 = 128 bins重叠
                 bin_size: int = 8192,        # 8KB per bin
                 chr_start: int = 0,          
                 chr_end: Optional[int] = None):
        
        self.chr_name = chr_name
        self.model_path = model_path
        self.seq_path = seq_path
        self.chip_path = chip_path
        self.celltype = celltype
        self.output_path = output_path
        self.window_size = window_size
        self.step_size = step_size
        self.bin_size = bin_size
        self.chr_start = chr_start
        self.chr_end = chr_end
        
        # 计算每个窗口的bin数量
        self.bins_per_window = window_size // bin_size  # 应该是256
        self.step_bins = step_size // bin_size  # 应该是128
        
        print(f"窗口配置: {window_size//1024}KB窗口, {self.bins_per_window}个bins, {bin_size}bp/bin")
        
        # 创建输出目录
        self.save_path = f'{output_path}/{celltype}/whole_chr_analysis/{chr_name}'
        for subdir in ['stage1_coarse', 'stage2_fine', 'gram_results', 'impact_scores', 
                      'visualizations', 'candidates', 'reports']:
            os.makedirs(f'{self.save_path}/{subdir}', exist_ok=True)
        
        # 加载模型和数据
        self._load_model_and_data()
        self._calculate_windows()
    
    def _load_model_and_data(self):
        """加载模型和数据"""
        print(f"加载模型: {self.model_path}")
        self.model = model_utils.load_default(self.model_path)
        self.model.eval()
        
        print(f"加载全染色体数据: {self.chr_name}")
        seq_file_path = os.path.join(self.seq_path, f'{self.chr_name}.fa.gz')
        self.seq, self.chip = infer.load_data_default(self.chr_name, seq_file_path, self.chip_path)
        
        if self.chr_end is None:
            self.chr_end = len(self.seq)
        
        print(f"染色体{self.chr_name}长度: {self.chr_end:,} bp")
    
    def _calculate_windows(self):
        """计算扫描窗口"""
        self.windows = []
        current_pos = self.chr_start
        
        while current_pos + self.window_size <= self.chr_end:
            self.windows.append(current_pos)
            current_pos += self.step_size
        
        print(f"总共需要扫描 {len(self.windows)} 个窗口")
        print(f"预计覆盖范围: {self.chr_start:,} - {self.windows[-1] + self.window_size:,} bp")
    
    def stage1_coarse_screening(self, target_layers: List[str] = None) -> Tuple[Dict[str, List[np.ndarray]], List[int], List[float], List[int], List[int]]:
        """
        第一阶段：8KB bin级别的粗筛
        返回：GRAM结果，窗口位置，Impact Scores，deletion位置
        """
        if target_layers is None:
            target_layers = ['encoder.encoder_seq', 'encoder.encoder_epi', 'encoder.gate_fusion']
        
        print(f"\n=== 第一阶段：8KB bin级别粗筛 ===")
        print(f"目标层: {target_layers}")
        print(f"每个窗口将进行 {self.bins_per_window} 次deletion测试")
        
        # 存储结果
        all_gram_results = {layer: [] for layer in target_layers}
        all_window_positions = []
        all_impact_scores = []
        all_deletion_starts = []
        all_deletion_ends = []
        
        # 创建GRAM分析器
        gram_analyzer = MultiLayerGRAMAnalyzer(self.model, target_layers)
        
        try:
            for i, window_start in enumerate(tqdm(self.windows, desc="Stage 1 - 粗筛")):
                try:
                    # === GRAM分析 ===
                    # 加载当前窗口的数据
                    seq_region, chip_region = infer.get_data_at_interval(
                        self.chr_name, window_start, window_start + self.window_size, 
                        self.seq, self.chip)
                    
                    # 预处理输入并计算GRAM
                    inputs = infer.preprocess_default(seq_region, chip_region)
                    gram_results = gram_analyzer.compute_gram_for_all_layers(inputs)
                    
                    # 存储GRAM结果
                    window_gram = {}
                    for layer_name, (gram_1d, _) in gram_results.items():
                        gram_np = gram_1d.detach().cpu().numpy()
                        # 确保GRAM数组长度是256
                        if len(gram_np) != self.bins_per_window:
                            print(f"警告: GRAM长度 {len(gram_np)} != {self.bins_per_window}")
                            # 调整到正确长度
                            if len(gram_np) > self.bins_per_window:
                                gram_np = gram_np[:self.bins_per_window]
                            else:
                                gram_np = np.pad(gram_np, (0, self.bins_per_window - len(gram_np)))
                        
                        all_gram_results[layer_name].append(gram_np)
                        window_gram[layer_name] = gram_np
                    
                    all_window_positions.append(window_start)
                    
                    # === Impact Score分析 ===
                    # 对当前窗口的每个bin进行deletion测试
                    window_impact_scores = []
                    window_deletion_starts = []
                    window_deletion_ends = []
                    
                    for bin_idx in range(self.bins_per_window):
                        deletion_start = window_start + bin_idx * self.bin_size
                        deletion_end = deletion_start + self.bin_size
                        
                        try:
                            # 计算deletion对Hi-C矩阵的影响
                            pred_start = window_start  # 预测窗口就是当前2MB窗口
                            
                            pred, pred_deletion, diff_map = screening.predict_difference(
                                self.chr_name, pred_start, deletion_start, self.bin_size,
                                self.model, self.seq, self.chip)
                            
                            # 计算Impact Score
                            impact_score = np.mean(np.abs(diff_map))
                            
                            window_impact_scores.append(impact_score)
                            window_deletion_starts.append(deletion_start)
                            window_deletion_ends.append(deletion_end)
                            
                        except Exception as e:
                            print(f"Bin {bin_idx} deletion失败: {e}")
                            # 填充默认值
                            window_impact_scores.append(0.0)
                            window_deletion_starts.append(deletion_start)
                            window_deletion_ends.append(deletion_end)
                    
                    # 验证GRAM和Impact Score的对齐
                    assert len(window_impact_scores) == self.bins_per_window, \
                        f"Impact scores长度不匹配: {len(window_impact_scores)} != {self.bins_per_window}"
                    
                    # 添加到总列表
                    all_impact_scores.extend(window_impact_scores)
                    all_deletion_starts.extend(window_deletion_starts)
                    all_deletion_ends.extend(window_deletion_ends)
                    
                    # 验证空间对齐
                    if i == 0:  # 只在第一个窗口验证
                        print(f"空间对齐验证:")
                        print(f"  GRAM点数: {len(window_gram[target_layers[0]])}")
                        print(f"  Impact Score点数: {len(window_impact_scores)}")
                        print(f"  每个点对应: {self.bin_size}bp")
                        print(f"  窗口范围: {window_start:,} - {window_start + self.window_size:,}")
                        print(f"  第一个bin: {window_deletion_starts[0]:,} - {window_deletion_ends[0]:,}")
                        print(f"  最后一个bin: {window_deletion_starts[-1]:,} - {window_deletion_ends[-1]:,}")
                    
                    # 保存当前窗口结果
                    window_results = {
                        'window_start': window_start,
                        'gram_results': window_gram,
                        'impact_scores': window_impact_scores,
                        'deletion_starts': window_deletion_starts,
                        'deletion_ends': window_deletion_ends
                    }
                    
                    # 保存窗口级别的结果
                    np.savez(f'{self.save_path}/stage1_coarse/window_{i:04d}_{window_start}.npz', 
                            **window_results)
                    
                    # 定期保存中间结果
                    if (i + 1) % 10 == 0:
                        print(f"已完成 {i + 1}/{len(self.windows)} 个窗口")
                        self._save_stage1_checkpoint(all_gram_results, all_window_positions,
                                                   all_impact_scores, all_deletion_starts, 
                                                   all_deletion_ends, i + 1)
                
                except Exception as e:
                    print(f"窗口 {window_start} 处理失败: {e}")
                    # 填充空结果以保持对齐
                    for layer_name in target_layers:
                        all_gram_results[layer_name].append(np.zeros(self.bins_per_window))
                    all_window_positions.append(window_start)
                    # 为每个bin添加零值
                    for bin_idx in range(self.bins_per_window):
                        all_impact_scores.append(0.0)
                        all_deletion_starts.append(window_start + bin_idx * self.bin_size)
                        all_deletion_ends.append(window_start + (bin_idx + 1) * self.bin_size)
                    continue
        
        finally:
            gram_analyzer.cleanup()
        
        # 保存最终结果
        self._save_stage1_final_results(all_gram_results, all_window_positions,
                                       all_impact_scores, all_deletion_starts, all_deletion_ends)
        
        return all_gram_results, all_window_positions, all_impact_scores, all_deletion_starts, all_deletion_ends
    
    def merge_overlapping_gram_results(self, all_gram_results: Dict[str, List[np.ndarray]], 
                                     all_window_positions: List[int]) -> Dict[str, np.ndarray]:
        """
        合并重叠区域的GRAM结果
        由于步长是1MB（128 bins），会有128 bins的重叠
        """
        print(f"\n=== 合并重叠区域的GRAM结果 ===")
        
        merged_results = {}
        
        for layer_name, gram_list in all_gram_results.items():
            if not gram_list:
                continue
                
            print(f"合并 {layer_name} 的结果...")
            
            # 计算总的bin数量
            total_bins = (all_window_positions[-1] - all_window_positions[0]) // self.bin_size + self.bins_per_window
            
            # 创建合并数组和计数数组
            merged_gram = np.zeros(total_bins)
            count_array = np.zeros(total_bins)
            
            # 合并所有窗口的结果
            for pos, gram_values in zip(all_window_positions, gram_list):
                start_bin = (pos - all_window_positions[0]) // self.bin_size
                end_bin = start_bin + len(gram_values)
                
                if end_bin <= len(merged_gram):
                    merged_gram[start_bin:end_bin] += gram_values
                    count_array[start_bin:end_bin] += 1
            
            # 对重叠区域取平均
            mask = count_array > 0
            merged_gram[mask] /= count_array[mask]
            
            merged_results[layer_name] = merged_gram
            
            print(f"{layer_name} 合并完成，最终长度: {len(merged_gram)} bins")
            print(f"  覆盖范围: {len(merged_gram) * self.bin_size:,} bp")
            print(f"  重叠区域数: {np.sum(count_array > 1)} bins")
        
        # 保存合并结果
        np.savez(f'{self.save_path}/gram_results/merged_gram_results.npz', **merged_results)
        
        return merged_results
    
    def identify_stage1_candidates(self, impact_scores: List[float], deletion_starts: List[int],
                                 deletion_ends: List[int], merged_gram_results: Dict[str, np.ndarray], 
                                 top_percentile: float = 1.0) -> pd.DataFrame:
        """识别第一阶段候选区域"""
        print(f"\n=== 识别第一阶段候选区域 (top {top_percentile}%) ===")
        
        # 计算阈值
        impact_array = np.array(impact_scores)
        threshold = np.percentile(impact_array, 100 - top_percentile)
        candidate_mask = impact_array >= threshold
        
        candidate_indices = np.where(candidate_mask)[0]
        print(f"找到 {len(candidate_indices)} 个候选bins")
        
        # 获取序列和表观遗传学GRAM
        seq_gram = merged_gram_results.get('encoder.encoder_seq', np.array([]))
        epi_gram = merged_gram_results.get('encoder.encoder_epi', np.array([]))
        
        # 创建候选区域列表
        candidates = []
        
        for idx in candidate_indices:
            deletion_start = deletion_starts[idx]
            deletion_end = deletion_ends[idx]
            impact_score = impact_scores[idx]
            
            # 计算对应的GRAM值
            # idx直接对应merged GRAM中的位置（考虑到起始位置偏移）
            gram_idx = (deletion_start - deletion_starts[0]) // self.bin_size
            
            if 0 <= gram_idx < len(seq_gram):
                seq_value = seq_gram[gram_idx]
            else:
                seq_value = 0
                
            if 0 <= gram_idx < len(epi_gram):
                epi_value = epi_gram[gram_idx]
            else:
                epi_value = 0
            
            # 分类（基于75分位数）
            seq_threshold = np.percentile(seq_gram, 75) if len(seq_gram) > 0 else 0
            epi_threshold = np.percentile(epi_gram, 75) if len(epi_gram) > 0 else 0
            
            if seq_value >= seq_threshold and epi_value >= epi_threshold:
                category = "Both_High"
            elif seq_value >= seq_threshold and epi_value < epi_threshold:
                category = "Seq_High_Epi_Low"
            elif seq_value < seq_threshold and epi_value >= epi_threshold:
                category = "Seq_Low_Epi_High"
            else:
                category = "Both_Low"
            
            candidates.append({
                'chr': self.chr_name,
                'start': deletion_start,
                'end': deletion_end,
                'bin_idx': gram_idx,
                'impact_score': impact_score,
                'seq_gram': seq_value,
                'epi_gram': epi_value,
                'category': category
            })
        
        # 创建DataFrame
        candidates_df = pd.DataFrame(candidates)
        
        # 按category分组统计
        if len(candidates_df) > 0:
            category_counts = candidates_df['category'].value_counts()
            print("\n第一阶段候选区域分类统计:")
            for category, count in category_counts.items():
                print(f"  {category}: {count} 个bins")
        
        # 保存结果
        candidates_df.to_csv(f'{self.save_path}/candidates/stage1_candidates_top_{top_percentile}percent.csv', 
                           index=False)
        
        return candidates_df
    
    def stage2_fine_screening(self, candidates_df: pd.DataFrame, 
                            extend_region: int = 4096,  # 在候选bin前后各扩展4KB
                            fine_deletion_size: int = 1024) -> pd.DataFrame:
        """
        第二阶段：在候选区域进行1KB精细扫描
        """
        print(f"\n=== 第二阶段：1KB精细扫描 ===")
        print(f"候选区域数: {len(candidates_df)}")
        print(f"精细deletion大小: {fine_deletion_size}bp")
        print(f"扩展区域: ±{extend_region}bp")
        
        fine_results = []
        
        for i, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Stage 2 - 精细扫描"):
            try:
                # 定义精细扫描区域
                candidate_start = row['start']
                candidate_end = row['end']
                scan_start = max(0, candidate_start - extend_region)
                scan_end = candidate_end + extend_region
                scan_length = scan_end - scan_start
                
                # 计算需要进行的精细deletion次数
                num_fine_deletions = scan_length // fine_deletion_size
                
                print(f"\n处理候选区域 {i+1}/{len(candidates_df)}: {candidate_start:,}-{candidate_end:,}")
                print(f"  扫描区域: {scan_start:,}-{scan_end:,} ({scan_length:,}bp)")
                print(f"  精细deletion次数: {num_fine_deletions}")
                
                # 为精细扫描准备预测窗口（以候选区域为中心的2MB窗口）
                pred_center = (candidate_start + candidate_end) // 2
                pred_start = pred_center - self.window_size // 2
                
                # 对扫描区域内每个1KB进行deletion测试
                fine_impact_scores = []
                fine_deletion_positions = []
                
                for j in range(num_fine_deletions):
                    deletion_start = scan_start + j * fine_deletion_size
                    deletion_end = deletion_start + fine_deletion_size
                    
                    try:
                        # 计算1KB deletion的影响
                        pred, pred_deletion, diff_map = screening.predict_difference(
                            self.chr_name, pred_start, deletion_start, fine_deletion_size,
                            self.model, self.seq, self.chip)
                        
                        fine_impact_score = np.abs(diff_map.mean())
                        fine_impact_scores.append(fine_impact_score)
                        fine_deletion_positions.append(deletion_start)
                        
                    except Exception as e:
                        print(f"    精细deletion {deletion_start:,} 失败: {e}")
                        fine_impact_scores.append(0.0)
                        fine_deletion_positions.append(deletion_start)
                
                # 找到精细扫描中的最高影响位点
                if fine_impact_scores:
                    max_idx = np.argmax(fine_impact_scores)
                    max_impact = fine_impact_scores[max_idx]
                    max_position = fine_deletion_positions[max_idx]
                    
                    # 记录精细扫描结果
                    fine_result = {
                        'original_bin_start': candidate_start,
                        'original_bin_end': candidate_end,
                        'original_impact_score': row['impact_score'],
                        'original_category': row['category'],
                        'scan_start': scan_start,
                        'scan_end': scan_end,
                        'num_fine_tests': len(fine_impact_scores),
                        'max_fine_impact': max_impact,
                        'max_fine_position': max_position,
                        'max_fine_end': max_position + fine_deletion_size,
                        'fine_improvement': max_impact / row['impact_score'] if row['impact_score'] > 0 else 1.0,
                        'all_fine_scores': fine_impact_scores,
                        'all_fine_positions': fine_deletion_positions
                    }
                    
                    fine_results.append(fine_result)
                    
                    print(f"  最大精细影响: {max_impact:.4f} @ {max_position:,}-{max_position + fine_deletion_size:,}")
                    print(f"  相对原始影响提升: {fine_result['fine_improvement']:.2f}x")
                
                # 保存当前候选区域的详细结果
                region_save_path = f'{self.save_path}/stage2_fine/candidate_{i:04d}_{candidate_start}_{candidate_end}.npz'
                np.savez(region_save_path,
                        scan_start=scan_start,
                        scan_end=scan_end,
                        fine_impact_scores=fine_impact_scores,
                        fine_deletion_positions=fine_deletion_positions,
                        original_info=row.to_dict())
                
            except Exception as e:
                print(f"候选区域 {candidate_start:,}-{candidate_end:,} 精细扫描失败: {e}")
                continue
        
        # 创建精细结果DataFrame
        fine_results_df = pd.DataFrame(fine_results)
        
        if len(fine_results_df) > 0:
            # 保存精细扫描结果
            fine_results_df.to_csv(f'{self.save_path}/candidates/stage2_fine_results.csv', index=False)
            
            # 统计精细扫描效果
            print(f"\n第二阶段精细扫描完成:")
            print(f"  处理候选区域: {len(fine_results_df)}")
            print(f"  平均提升倍数: {fine_results_df['fine_improvement'].mean():.2f}x")
            print(f"  最大提升倍数: {fine_results_df['fine_improvement'].max():.2f}x")
            
            # Top精细定位结果
            top_fine = fine_results_df.nlargest(10, 'max_fine_impact')
            print(f"\nTop 10 精细定位结果:")
            for i, (_, row) in enumerate(top_fine.iterrows(), 1):
                print(f"  {i:2d}. {self.chr_name}:{row['max_fine_position']:,}-{row['max_fine_end']:,} "
                      f"(Impact: {row['max_fine_impact']:.4f}, "
                      f"类别: {row['original_category']})")
        
        return fine_results_df
    
    def _save_stage1_checkpoint(self, all_gram_results, all_window_positions, 
                               all_impact_scores, all_deletion_starts, all_deletion_ends, checkpoint):
        """保存第一阶段检查点"""
        checkpoint_path = f'{self.save_path}/stage1_coarse/checkpoint_{checkpoint}'
        os.makedirs(checkpoint_path, exist_ok=True)
        
        np.savez(f'{checkpoint_path}/checkpoint_data.npz',
                gram_results={k: np.array(v, dtype=object) for k, v in all_gram_results.items()},
                window_positions=all_window_positions,
                impact_scores=all_impact_scores,
                deletion_starts=all_deletion_starts,
                deletion_ends=all_deletion_ends)
    
    def _save_stage1_final_results(self, all_gram_results, all_window_positions,
                                  all_impact_scores, all_deletion_starts, all_deletion_ends):
        """保存第一阶段最终结果"""
        # 保存GRAM结果
        for layer_name, gram_list in all_gram_results.items():
            np.save(f'{self.save_path}/gram_results/{layer_name}_all_windows.npy', 
                   np.array(gram_list, dtype=object))
        
        np.save(f'{self.save_path}/gram_results/all_window_positions.npy', np.array(all_window_positions))
        
        # 保存Impact Score结果
        np.save(f'{self.save_path}/impact_scores/all_impact_scores.npy', np.array(all_impact_scores))
        np.save(f'{self.save_path}/impact_scores/all_deletion_starts.npy', np.array(all_deletion_starts))
        np.save(f'{self.save_path}/impact_scores/all_deletion_ends.npy', np.array(all_deletion_ends))
        
        # 保存为bedgraph格式
        df = pd.DataFrame({
            'chr': self.chr_name,
            'start': all_deletion_starts,
            'end': all_deletion_ends,
            'score': all_impact_scores
        })
        df.to_csv(f'{self.save_path}/impact_scores/{self.chr_name}_stage1_impact_scores.bedgraph', 
                 sep='\t', index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Whole Chromosome Analysis')
    
    # 基本参数
    parser.add_argument('--chr', dest='chr_name', required=True,
                        help='染色体名称 (例如: chr21)')
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
    
    # 第一阶段参数
    parser.add_argument('--window-size', dest='window_size', type=int, default=2097152,
                        help='扫描窗口大小 (bp), 默认: 2MB')
    parser.add_argument('--step-size', dest='step_size', type=int, default=1048576,
                        help='扫描步长 (bp), 默认: 1MB') 
    parser.add_argument('--bin-size', dest='bin_size', type=int, default=8192,
                        help='Bin大小 (bp), 默认: 8KB')
    parser.add_argument('--chr-start', dest='chr_start', type=int, default=0,
                        help='染色体起始位置 (bp)')
    parser.add_argument('--chr-end', dest='chr_end', type=int, default=None,
                        help='染色体结束位置 (bp)')
    
    # 第二阶段参数
    parser.add_argument('--top-percent', dest='top_percent', type=float, default=1.0,
                        help='进入第二阶段的候选区域百分比 (%)')
    parser.add_argument('--extend-region', dest='extend_region', type=int, default=4096,
                        help='第二阶段扩展区域大小 (bp), 默认: 4KB')
    parser.add_argument('--fine-deletion-size', dest='fine_deletion_size', type=int, default=1024,
                        help='第二阶段精细deletion大小 (bp), 默认: 1KB')
    
    # 分析参数
    parser.add_argument('--target-layers', dest='target_layers', nargs='+',
                        default=['encoder.encoder_seq', 'encoder.encoder_epi', 'encoder.gate_fusion'],
                        help='GRAM分析的目标层')
    
    # 功能开关
    parser.add_argument('--skip-stage1', dest='skip_stage1', action='store_true',
                        help='跳过第一阶段，从已有结果开始')
    parser.add_argument('--skip-stage2', dest='skip_stage2', action='store_true',
                        help='跳过第二阶段精细扫描')
    parser.add_argument('--stage1-only', dest='stage1_only', action='store_true',
                        help='仅运行第一阶段')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("两阶段全染色体分析")
    print("=" * 80)
    print(f"染色体: {args.chr_name}")
    print(f"细胞类型: {args.celltype}")
    print(f"第一阶段: {args.window_size//1024}KB窗口, {args.step_size//1024}KB步长, {args.bin_size}bp bins")
    print(f"第二阶段: 前{args.top_percent}%候选区域, {args.fine_deletion_size}bp精细deletion")
    print("=" * 80)
    
    try:
        # 创建分析器
        analyzer = WholeChromosomeAnalyzer(
            chr_name=args.chr_name,
            model_path=args.model_path,
            seq_path=args.seq_path,
            chip_path=args.chip_path,
            celltype=args.celltype,
            output_path=args.output_path,
            window_size=args.window_size,
            step_size=args.step_size,
            bin_size=args.bin_size,
            chr_start=args.chr_start,
            chr_end=args.chr_end
        )
        
        # 第一阶段：8KB bin级别粗筛
        if not args.skip_stage1:
            print("\n第一阶段：8KB bin级别粗筛...")
            all_gram_results, all_window_positions, all_impact_scores, all_deletion_starts, all_deletion_ends = \
                analyzer.stage1_coarse_screening(args.target_layers)
        else:
            print("\n跳过第一阶段，加载已有结果...")
            # 加载已有结果的代码
            # ... (实现加载逻辑)
        
        # 合并重叠结果
        print("\n合并重叠区域...")
        merged_gram_results = analyzer.merge_overlapping_gram_results(all_gram_results, all_window_positions)
        
        # 识别候选区域
        print("\n识别候选区域...")
        candidates_df = analyzer.identify_stage1_candidates(
            all_impact_scores, all_deletion_starts, all_deletion_ends, 
            merged_gram_results, args.top_percent)
        
        if args.stage1_only:
            print("\n仅运行第一阶段，结束。")
            return
        
        # 第二阶段：1KB精细扫描
        if not args.skip_stage2 and len(candidates_df) > 0:
            print("\n第二阶段：1KB精细扫描...")
            fine_results_df = analyzer.stage2_fine_screening(
                candidates_df, args.extend_region, args.fine_deletion_size)
        else:
            print("\n跳过第二阶段或无候选区域")
            fine_results_df = pd.DataFrame()
        
        # 生成报告
        print("\n生成分析报告...")
        generate_two_stage_report(analyzer.save_path, args, merged_gram_results, 
                                all_impact_scores, candidates_df, fine_results_df)
        
        print("\n" + "=" * 80)
        print("两阶段分析完成!")
        print(f"结果保存在: {analyzer.save_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

def generate_two_stage_report(save_path: str, args, merged_gram_results: Dict[str, np.ndarray],
                            all_impact_scores: List[float], candidates_df: pd.DataFrame, 
                            fine_results_df: pd.DataFrame):
    """生成两阶段分析报告"""
    
    report_path = f'{save_path}/reports/two_stage_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("两阶段全染色体分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"分析参数:\n")
        f.write(f"  染色体: {args.chr_name}\n")
        f.write(f"  细胞类型: {args.celltype}\n")
        f.write(f"  第一阶段窗口: {args.window_size:,}bp ({args.window_size//args.bin_size} bins)\n")
        f.write(f"  Bin大小: {args.bin_size:,}bp\n")
        f.write(f"  步长: {args.step_size:,}bp\n")
        f.write(f"  候选区域阈值: 前{args.top_percent}%\n")
        f.write(f"  第二阶段精细deletion: {args.fine_deletion_size:,}bp\n\n")
        
        # 第一阶段结果
        f.write("第一阶段结果 (8KB bin级别筛选):\n")
        f.write("=" * 50 + "\n")
        
        impact_array = np.array(all_impact_scores)
        f.write(f"总扫描bins: {len(all_impact_scores):,}\n")
        f.write(f"覆盖范围: {len(all_impact_scores) * args.bin_size:,}bp\n")
        f.write(f"Impact Score范围: [{impact_array.min():.4f}, {impact_array.max():.4f}]\n")
        f.write(f"Impact Score均值: {impact_array.mean():.4f} ± {impact_array.std():.4f}\n")
        
        # GRAM统计
        for layer_name, gram_values in merged_gram_results.items():
            f.write(f"\n{layer_name} GRAM:\n")
            f.write(f"  覆盖bins: {len(gram_values):,}\n")
            f.write(f"  值范围: [{gram_values.min():.4f}, {gram_values.max():.4f}]\n")
            f.write(f"  均值: {gram_values.mean():.4f} ± {gram_values.std():.4f}\n")
        
        # 候选区域统计
        f.write(f"\n识别候选区域: {len(candidates_df)} bins\n")
        if len(candidates_df) > 0:
            f.write("候选区域分类:\n")
            category_counts = candidates_df['category'].value_counts()
            for category, count in category_counts.items():
                f.write(f"  {category}: {count} bins\n")
        
        # 第二阶段结果
        if len(fine_results_df) > 0:
            f.write(f"\n第二阶段结果 (1KB精细定位):\n")
            f.write("=" * 50 + "\n")
            f.write(f"精细扫描候选区域: {len(fine_results_df)}\n")
            f.write(f"平均精细度提升: {fine_results_df['fine_improvement'].mean():.2f}x\n")
            f.write(f"最大精细度提升: {fine_results_df['fine_improvement'].max():.2f}x\n")
            
            # Top精细定位结果
            f.write(f"\nTop 10 精细定位调控元件:\n")
            top_fine = fine_results_df.nlargest(10, 'max_fine_impact')
            for i, (_, row) in enumerate(top_fine.iterrows(), 1):
                f.write(f"{i:2d}. {args.chr_name}:{row['max_fine_position']:,}-{row['max_fine_end']:,} ")
                f.write(f"(Impact: {row['max_fine_impact']:.4f}, ")
                f.write(f"类别: {row['original_category']}, ")
                f.write(f"提升: {row['fine_improvement']:.1f}x)\n")
        
        f.write(f"\n结果文件:\n")
        f.write(f"  第一阶段候选: {save_path}/candidates/stage1_candidates_top_{args.top_percent}percent.csv\n")
        if len(fine_results_df) > 0:
            f.write(f"  第二阶段精细结果: {save_path}/candidates/stage2_fine_results.csv\n")
        f.write(f"  GRAM结果: {save_path}/gram_results/\n")
        f.write(f"  Impact Score: {save_path}/impact_scores/\n")
    
    print(f"详细分析报告保存到: {report_path}")

if __name__ == "__main__":
    main() 
