import os
import numpy as np
import pandas as pd
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import plot_utils


# ===================== Insulation Score 计算 =====================
def chr_score(matrix, res=10000, radius=500000, pseudocount_coeff=30):
    """
    计算输入矩阵的 insulation score 列表
    """
    matrix = np.asarray(matrix, dtype=float)
    pseudocount = matrix.mean() * pseudocount_coeff
    if pseudocount == 0:
        pseudocount = 1e-6
    pixel_radius = int(radius / res)
    scores = []
    for loc in range(len(matrix)):
        scores.append(point_score(loc, pixel_radius, matrix, pseudocount))
    return np.array(scores)


def point_score(locus, radius, matrix, pseudocount):
    """
    计算单个位置的 insulation score：
    score = (max(左上/右下子矩阵的均值) + pseudocount) / (中间子矩阵均值 + pseudocount)
    """
    l_edge = max(locus - radius, 0)
    r_edge = min(locus + radius, len(matrix))
    l_mask = matrix[l_edge:locus, l_edge:locus]
    r_mask = matrix[locus:r_edge, locus:r_edge]
    center_mask = matrix[l_edge:locus, locus:r_edge]

    left_mean = l_mask.mean() if l_mask.size else 0.0
    right_mean = r_mask.mean() if r_mask.size else 0.0
    center_mean = center_mask.mean() if center_mask.size else 0.0

    numerator = max(left_mean, right_mean) + pseudocount
    denominator = center_mean + pseudocount
    if denominator == 0:
        denominator = 1e-6
    return numerator / denominator
# ================================================================= 

def main():
    parser = argparse.ArgumentParser(description='Hi-READ Deletion Module with Visualization.')
    
    # Output location
    parser.add_argument('--out', dest='output_path', 
                        default='outputs',
                        help='output path for storing results (default: %(default)s)')

    # Location related params
    parser.add_argument('--celltype', dest='celltype', 
                        help='Sample cell type for prediction, used for output separation', required=True)
    parser.add_argument('--chr', dest='chr_name', 
                        help='Chromosome for prediction', required=True)
    parser.add_argument('--start', dest='start', type=int,
                        help='Starting point for prediction (width is 2097152 bp which is the input window size)', required=True)
    parser.add_argument('--model', dest='model_path', 
                        help='Path to the model checkpoint', required=True)
    parser.add_argument('--seq', dest='seq_path', 
                        help='Path to the folder where the sequence .fa.gz files are stored', required=True)
    parser.add_argument('--chip', dest='chip_path', 
                        help='Path to the folder where the ChIP-seq .bw files are stored', required=True)

    # Deletion related params
    parser.add_argument('--del-start', dest='deletion_start', type=int,
                        help='Starting point for deletion.', required=True)
    parser.add_argument('--del-width', dest='deletion_width', type=int,
                        help='Width for deletion.', required=True)
    parser.add_argument('--padding', dest='end_padding_type', 
                        default='zero',
                        help='Padding type, either zero or follow. Using zero: the missing region at the end will be padded with zero for chip and atac seq, while sequence will be padded with N (unknown necleotide). Using follow: the end will be padded with features in the following region (default: %(default)s)')
    parser.add_argument('--hide-line', dest='hide_deletion_line', 
                        action = 'store_true',
                        help='Remove the line showing deletion site (default: %(default)s)')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    single_deletion(args.output_path, args.celltype, args.chr_name, args.start, 
                    args.deletion_start, args.deletion_width, 
                    args.model_path,
                    args.seq_path, args.chip_path, 
                    show_deletion_line = not args.hide_deletion_line,
                    end_padding_type = args.end_padding_type)

def single_deletion(output_path, celltype, chr_name, start, deletion_start, deletion_width, model_path, seq_path, chip_path, show_deletion_line = True, end_padding_type = 'zero'):
    """
    Perform single deletion prediction and visualization with comparison plots
    """
    
    # 构建完整的序列文件路径（与其他脚本保持一致）
    seq_file_path = os.path.join(seq_path, f'{chr_name}.fa.gz')
    print(f"使用序列文件: {seq_file_path}")
    
    # Load original region for prediction (standard window)
    seq_region_original, chip_region_original = infer.load_region(chr_name, 
            start, seq_file_path, chip_path, window = 2097152)
    
    # Load extended region that accommodates deletion
    window = 2097152 + deletion_width
    seq_region, chip_region = infer.load_region(chr_name, 
            start, seq_file_path, chip_path, window = window)
    
    # Create deleted version
    seq_region_deleted, chip_region_deleted = deletion_with_padding(start, 
            deletion_start, deletion_width, seq_region, chip_region, 
             end_padding_type)
    
    # Predictions
    print("Predicting original region...")
    pred_original = infer.prediction(seq_region_original, chip_region_original, model_path)
    
    print("Predicting deleted region...")
    pred_deleted = infer.prediction(seq_region_deleted, chip_region_deleted, model_path)
    
    # 计算删除区域在 heatmap 中的像素位置
    bin_size = 8192  # 2097152 / 256 = 8192 bp per bin
    deletion_bin_start = int((deletion_start - start) / bin_size)
    deletion_bin_width = int(deletion_width / bin_size)
    deletion_bin_end = deletion_bin_start + deletion_bin_width
    
    # ============ 创建对齐后的删除矩阵 ============
    # 将删除后的矩阵重新排列，使基因组坐标对齐
    print("Creating aligned deletion matrix...")
    pred_deleted_aligned = create_aligned_deletion_matrix(
        pred_deleted, pred_original, deletion_bin_start, deletion_bin_width
    )
    
    # Calculate difference (现在是对齐后的比较)
    diff_map = pred_deleted_aligned - pred_original
    
    # ============ 计算 Insulation Score ============
    print("Calculating insulation scores...")
    insulation_original = chr_score(pred_original, res=10000, radius=500000, pseudocount_coeff=30)
    # 对齐后的 insulation score
    insulation_deleted_aligned = create_aligned_insulation(
        chr_score(pred_deleted, res=10000, radius=500000, pseudocount_coeff=30),
        deletion_bin_start, deletion_bin_width, len(insulation_original)
    )
    insulation_diff = insulation_deleted_aligned - insulation_original
    
    # Create comparison plot with insulation score tracks
    # 使用对齐后的矩阵进行可视化
    plot = MatrixPlotDeletion(
        output_path, pred_original, pred_deleted_aligned, diff_map, 'deletion',
        celltype, chr_name, start, deletion_start, deletion_width,
        insulation_original=insulation_original,
        insulation_deleted=insulation_deleted_aligned,
        insulation_diff=insulation_diff,
        deletion_bin_start=deletion_bin_start,
        deletion_bin_end=deletion_bin_end,
        padding_type=end_padding_type, show_deletion_line=show_deletion_line)
    plot.plot()

def deletion_with_padding(start, deletion_start, deletion_width, seq_region, chip_region, end_padding_type):
    ''' Delete all signals at a specfied location with corresponding padding at the end '''
    # end_padding_type takes values of either 'zero' or 'follow'
    if end_padding_type == 'zero':
        seq_region, chip_region = zero_region(seq_region, 
                chip_region)
    elif end_padding_type == 'follow': pass
    else: raise Exception('unknown padding')
    # Deletion
    seq_region, chip_region = delete(deletion_start - start, 
            deletion_start - start + deletion_width, 
            seq_region, chip_region)
    return seq_region, chip_region

def zero_region(seq, chip, window = 2097152):
    ''' Replace signal with zero. N for sequence and 0 for CHIP and ATAC '''
    seq[window:] = [0, 0, 0, 0, 1]
    chip[window:] = 0
    
    return seq, chip

def delete(start, end, seq, chip, window = 2097152):
    seq = np.delete(seq, np.s_[start:end], axis = 0)
    chip = np.delete(chip, np.s_[start:end])
    return seq[:window], chip[:window]


def create_aligned_deletion_matrix(pred_deleted, pred_original, deletion_bin_start, deletion_bin_width):
    """
    创建对齐后的删除矩阵：
    - 删除区域之前的部分：使用 pred_deleted 的对应区域
    - 删除区域本身：留空（NaN，后续显示为白色）
    - 删除区域之后的部分：从 pred_deleted 取数据，但放到正确的基因组位置
    
    这样 Before 和 After 矩阵的相同位置对应相同的基因组坐标
    """
    n = pred_original.shape[0]
    
    # 创建一个 NaN 填充的矩阵
    aligned = np.full_like(pred_original, np.nan)
    
    # 1. 删除区域之前的部分：直接复制
    # pred_deleted 的 [0:deletion_bin_start] 对应原始的 [0:deletion_bin_start]
    if deletion_bin_start > 0:
        aligned[:deletion_bin_start, :deletion_bin_start] = pred_deleted[:deletion_bin_start, :deletion_bin_start]
    
    # 2. 删除区域本身：保持 NaN（会显示为白色）
    
    # 3. 删除区域之后的部分：
    # pred_deleted 的 [deletion_bin_start:] 对应原始的 [deletion_bin_start + deletion_bin_width:]
    # 我们需要把 pred_deleted 的后半部分放到 aligned 的正确位置
    
    # pred_deleted 中删除后的区域从 deletion_bin_start 开始
    # 这些数据对应原始矩阵中 deletion_bin_start + deletion_bin_width 之后的区域
    
    after_start_in_deleted = deletion_bin_start
    after_start_in_original = deletion_bin_start + deletion_bin_width
    
    # 可用的数据长度
    available_length = n - after_start_in_deleted  # pred_deleted 中剩余的数据
    target_length = n - after_start_in_original     # aligned 中需要填充的空间
    
    if target_length > 0 and available_length > 0:
        copy_length = min(available_length, target_length)
        
        # 复制对角块（右下方区域）
        aligned[after_start_in_original:after_start_in_original+copy_length,
                after_start_in_original:after_start_in_original+copy_length] = \
            pred_deleted[after_start_in_deleted:after_start_in_deleted+copy_length,
                        after_start_in_deleted:after_start_in_deleted+copy_length]
        
        # 复制左上角与右下角的交叉区域
        # 左侧列（删除前区域与删除后区域的交互）
        if deletion_bin_start > 0:
            aligned[:deletion_bin_start, after_start_in_original:after_start_in_original+copy_length] = \
                pred_deleted[:deletion_bin_start, after_start_in_deleted:after_start_in_deleted+copy_length]
            aligned[after_start_in_original:after_start_in_original+copy_length, :deletion_bin_start] = \
                pred_deleted[after_start_in_deleted:after_start_in_deleted+copy_length, :deletion_bin_start]
    
    return aligned


def create_aligned_insulation(insulation_deleted, deletion_bin_start, deletion_bin_width, original_length):
    """
    创建对齐后的 insulation score：
    - 删除区域之前：使用原数据
    - 删除区域：NaN
    - 删除区域之后：平移到正确位置
    """
    aligned = np.full(original_length, np.nan)
    
    # 删除区域之前
    if deletion_bin_start > 0:
        aligned[:deletion_bin_start] = insulation_deleted[:deletion_bin_start]
    
    # 删除区域之后
    after_start_in_deleted = deletion_bin_start
    after_start_in_original = deletion_bin_start + deletion_bin_width
    
    available_length = len(insulation_deleted) - after_start_in_deleted
    target_length = original_length - after_start_in_original
    
    if target_length > 0 and available_length > 0:
        copy_length = min(available_length, target_length)
        aligned[after_start_in_original:after_start_in_original+copy_length] = \
            insulation_deleted[after_start_in_deleted:after_start_in_deleted+copy_length]
    
    return aligned

class MatrixPlotDeletion:
    """
    Custom plotting class for deletion comparison visualization
    With insulation score tracks below each heatmap
    """
    
    def __init__(self, output_path, pred_original, pred_deleted, diff_map, prefix, 
                 celltype, chr_name, start_pos, deletion_start, deletion_width,
                 insulation_original=None, insulation_deleted=None, insulation_diff=None,
                 deletion_bin_start=0, deletion_bin_end=0,
                 padding_type='zero', show_deletion_line=True):
        self.output_path = output_path
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.pred_original = pred_original
        self.pred_deleted = pred_deleted
        self.diff_map = diff_map
        self.insulation_original = insulation_original
        self.insulation_deleted = insulation_deleted
        self.insulation_diff = insulation_diff
        self.deletion_start = deletion_start
        self.deletion_width = deletion_width
        self.deletion_bin_start = deletion_bin_start
        self.deletion_bin_end = deletion_bin_end
        self.padding_type = padding_type
        self.show_deletion_line = show_deletion_line
        
        self.create_save_path(output_path, celltype, prefix)

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok=True)
        os.makedirs(f'{self.save_path}/npy', exist_ok=True)

    def get_colormap_hic(self):
        """Get colormap for Hi-C visualization"""
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        color_map.set_bad(color='lightgray')  # NaN 显示为浅灰色（删除区域）
        return color_map
    
    def get_colormap_diff(self):
        """Get colormap for difference visualization"""
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
        cmap.set_bad(color='lightgray')  # NaN 显示为浅灰色（删除区域）
        return cmap
    
    def rescale_coordinates(self, coords, zero_position):
        """将像素坐标转换为基因组坐标 (Mb)"""
        scaling_ratio = 8192  # 每个 bin 对应 8192 bp
        replaced_coords = coords * scaling_ratio + zero_position
        coords_mb = replaced_coords / 1000000
        str_list = [f'{item:.1f}' for item in coords_mb]
        return str_list

    def plot(self, vmin=0, vmax=3, diff_vmin=-0.5, diff_vmax=0.5):
        """Create the three-panel comparison plot with insulation score tracks"""
        plt.rcParams.update({'font.size': 10})
        
        # 创建 2x3 布局（上方 heatmap，下方 track）
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1], hspace=0.05, wspace=0.3)
        
        color_map_hic = self.get_colormap_hic()
        color_map_diff = self.get_colormap_diff()
        
        # 数据和标题
        matrices = [self.pred_original, self.pred_deleted, self.diff_map]
        insulations = [self.insulation_original, self.insulation_deleted, self.insulation_diff]
        titles = ['Before Deletion', 'After Deletion', 'Difference']
        cmaps = [color_map_hic, color_map_hic, color_map_diff]
        vmins = [vmin, vmin, diff_vmin]
        vmaxs = [vmax, vmax, diff_vmax]
        
        # 计算删除位置（像素坐标）
        scaling_factor = 256 / 2097152
        deletion_start_relative = self.deletion_start - self.start_pos
        deletion_start_px = deletion_start_relative * scaling_factor
        deletion_end_px = deletion_start_px + (self.deletion_width * scaling_factor)
        
        for col in range(3):
            # 上方：Heatmap
            ax_heatmap = fig.add_subplot(gs[0, col])
            im = ax_heatmap.imshow(matrices[col], cmap=cmaps[col], vmin=vmins[col], vmax=vmaxs[col], aspect='equal')
            ax_heatmap.set_title(titles[col], fontsize=12, fontweight='bold')
            
            # 添加删除线标记（红色虚线，细线）
            if self.show_deletion_line:
                ax_heatmap.axvline(x=deletion_start_px, color='red', linestyle='--', alpha=0.8, linewidth=1)
                ax_heatmap.axhline(y=deletion_start_px, color='red', linestyle='--', alpha=0.8, linewidth=1)
                ax_heatmap.axvline(x=deletion_end_px, color='red', linestyle='--', alpha=0.8, linewidth=1)
                ax_heatmap.axhline(y=deletion_end_px, color='red', linestyle='--', alpha=0.8, linewidth=1)
            
            # 设置刻度
            n_bins = len(matrices[col])
            tick_positions = np.arange(0, n_bins, 50)
            tick_labels = self.rescale_coordinates(tick_positions, self.start_pos)
            ax_heatmap.set_xticks(tick_positions)
            ax_heatmap.set_xticklabels([])  # 隐藏 x 轴标签（由下方 track 显示）
            ax_heatmap.set_yticks(tick_positions)
            ax_heatmap.set_yticklabels(tick_labels, fontsize=8)
            ax_heatmap.set_xlim(-0.5, n_bins - 0.5)
            ax_heatmap.set_ylim(n_bins - 0.5, -0.5)
            
            if col == 0:
                ax_heatmap.set_ylabel('Genomic position (Mb)', fontsize=10)
            
            # 添加 colorbar（使用 divider 确保不影响对齐）
            divider = make_axes_locatable(ax_heatmap)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=7)
            
            # 下方：Insulation Score Track
            if insulations[col] is not None:
                ax_track = fig.add_subplot(gs[1, col])
                x = np.arange(len(insulations[col]))
                
                # 绘制 insulation score，删除区域用红色
                if col < 2:  # Original 和 Deleted
                    mask_deletion = (x >= self.deletion_bin_start) & (x < self.deletion_bin_end)
                    
                    # 先画整体填充（浅色）
                    ax_track.fill_between(x, insulations[col], alpha=0.3, color='steelblue')
                    
                    # 删除区域用红色覆盖
                    if np.any(mask_deletion):
                        ax_track.fill_between(x, insulations[col], where=mask_deletion, 
                                             alpha=0.7, color='crimson')
                    
                    # 绘制线条
                    ax_track.plot(x, insulations[col], color='steelblue', linewidth=0.8)
                    
                    # 删除区域红色背景
                    ax_track.axvspan(self.deletion_bin_start, self.deletion_bin_end, 
                                    alpha=0.15, color='red')
                else:  # Difference
                    # 差异图用双色显示
                    ax_track.fill_between(x, insulations[col], where=insulations[col] >= 0, 
                                         alpha=0.5, color='steelblue', interpolate=True)
                    ax_track.fill_between(x, insulations[col], where=insulations[col] < 0, 
                                         alpha=0.5, color='crimson', interpolate=True)
                    ax_track.plot(x, insulations[col], color='gray', linewidth=0.8)
                    ax_track.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    # 删除区域背景
                    ax_track.axvspan(self.deletion_bin_start, self.deletion_bin_end, 
                                    alpha=0.15, color='red')
                
                # 严格对齐 heatmap 的 x 轴范围
                ax_track.set_xlim(-0.5, n_bins - 0.5)
                ax_track.spines['top'].set_visible(False)
                ax_track.spines['right'].set_visible(False)
                
                # 设置 x 轴刻度
                ax_track.set_xticks(tick_positions)
                ax_track.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')
                
                if col == 0:
                    ax_track.set_ylabel('Ins. Score', fontsize=9)
                elif col == 2:
                    ax_track.set_ylabel('Δ Ins. Score', fontsize=9)
                
                # 为 track 添加空白占位符以对齐 colorbar
                divider_track = make_axes_locatable(ax_track)
                cax_track = divider_track.append_axes("right", size="3%", pad=0.05)
                cax_track.axis('off')
        
        # 添加整体标题
        deletion_mb_start = self.deletion_start / 1000000
        deletion_mb_end = (self.deletion_start + self.deletion_width) / 1000000
        fig.suptitle(f'{self.chr_name}: Deletion {deletion_mb_start:.3f} - {deletion_mb_end:.3f} Mb '
                    f'(width: {self.deletion_width:,} bp)', fontsize=14, fontweight='bold', y=1.02)
        
        self.save_data(plt, fig)

    def save_data(self, plt, fig):
        """Save the plot and numerical data"""
        filename_base = f'{self.chr_name}_{self.start_pos}_deletion_{self.deletion_start}_{self.deletion_width}_{self.padding_type}'
        
        # Save plot
        fig.savefig(f'{self.save_path}/imgs/{filename_base}.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # Save numerical data
        np.save(f'{self.save_path}/npy/{filename_base}_original', self.pred_original)
        np.save(f'{self.save_path}/npy/{filename_base}_deleted', self.pred_deleted)
        np.save(f'{self.save_path}/npy/{filename_base}_difference', self.diff_map)
        
        # Save insulation scores
        if self.insulation_original is not None:
            np.save(f'{self.save_path}/npy/{filename_base}_insulation_original', self.insulation_original)
        if self.insulation_deleted is not None:
            np.save(f'{self.save_path}/npy/{filename_base}_insulation_deleted', self.insulation_deleted)
        if self.insulation_diff is not None:
            np.save(f'{self.save_path}/npy/{filename_base}_insulation_diff', self.insulation_diff)
        
        print(f"Results saved to {self.save_path}")
        print(f"  - Images: {filename_base}.png")
        print(f"  - Arrays: {filename_base}_original/deleted/difference/insulation_*.npy")

if __name__ == '__main__':
    main() 
