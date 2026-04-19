import os
import numpy as np
import pandas as pd
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import argparse
from skimage.transform import resize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.data import data_feature
from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import plot_utils

def main():
    parser = argparse.ArgumentParser(description='Hi-READ Translocation Module.')
    
    # Output location
    parser.add_argument('--out', dest='output_path', 
                        default='outputs',
                        help='output path for storing results (default: %(default)s)')

    # Location related params
    parser.add_argument('--celltype', dest='celltype', 
                        help='Sample cell type for prediction, used for output separation', required=True)
    
    # Translocation source (Chr7 in the example)
    parser.add_argument('--chr1', dest='chr1_name', 
                        help='First chromosome for translocation (e.g., chr7)', required=True)
    parser.add_argument('--pos1', dest='pos1', type=int,
                        help='Position on first chromosome for translocation breakpoint', required=True)
    
    # Translocation target (Chr9 in the example)
    parser.add_argument('--chr2', dest='chr2_name', 
                        help='Second chromosome for translocation (e.g., chr9)', required=True)
    parser.add_argument('--pos2', dest='pos2', type=int,
                        help='Position on second chromosome for translocation breakpoint', required=True)
    
    # Model and data paths
    parser.add_argument('--model', dest='model_path', 
                        help='Path to the model checkpoint', required=True)
    parser.add_argument('--seq', dest='seq_path', 
                        help='Path to the folder where the sequence .fa.gz files are stored', required=True)
    parser.add_argument('--chip', dest='chip_path', 
                        help='Path to the folder where the ChIP-seq .bw files are stored', required=True)
    
    # Ground truth Hi-C data path
    parser.add_argument('--hic', dest='hic_path', 
                        help='Path to the folder where the ground truth Hi-C matrix .npz files are stored', required=True)

    # Visualization options
    parser.add_argument('--hide-line', dest='hide_translocation_line', 
                        action='store_true',
                        help='Remove the line showing translocation breakpoints (default: %(default)s)')
    
    # Analysis mode selection
    parser.add_argument('--prediction-diff', dest='use_prediction_diff', 
                        action='store_true',
                        help='Use prediction difference mode: 4th panel shows concatenated predictions (Chr1 first half + Chr2 second half), 5th panel shows difference with translocation prediction (default: ground truth mode)')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    single_translocation(args.output_path, args.celltype, 
                        args.chr1_name, args.pos1, args.chr2_name, args.pos2,
                        args.model_path, args.seq_path, args.chip_path, args.hic_path,
                        show_translocation_line=not args.hide_translocation_line,
                        use_prediction_diff=args.use_prediction_diff)

def single_translocation(output_path, celltype, chr1_name, pos1, chr2_name, pos2, 
                        model_path, seq_path, chip_path, hic_path, show_translocation_line=True,
                        use_prediction_diff=False):
    """
    Perform single translocation prediction and visualization
    
    Parameters:
    - chr1_name: First chromosome (e.g., 'chr7')
    - pos1: Position on first chromosome for translocation breakpoint
    - chr2_name: Second chromosome (e.g., 'chr9')  
    - pos2: Position on second chromosome for translocation breakpoint
    - model_path: Path to the model checkpoint
    - seq_path: Path to sequence data
    - chip_path: Path to ChIP-seq data
    - hic_path: Path to ground truth Hi-C data
    - show_translocation_line: Whether to show breakpoint lines
    - use_prediction_diff: If True, use prediction difference mode (4th panel: concatenated predictions, 5th panel: prediction difference)
                          If False, use ground truth mode (4th panel: ground truth, 5th panel: GT-prediction difference)
    
    Creates fusion of chr1:pos1-1MB and chr2:pos2+1MB sequences
    """
    
    # Calculate start positions (pos - 1048576)
    start1 = max(0, pos1 - 1048576)  # Chr1 start position
    start2 = max(0, pos2 - 1048576)  # Chr2 start position
    
    print(f"Loading Chr1 region: {chr1_name}:{start1}-{pos1} (1MB before breakpoint)")
    print(f"Loading Chr2 region: {chr2_name}:{pos2}-{pos2 + 1048576} (1MB after breakpoint)")
    
    # Build full sequence file paths
    seq_file_path1 = os.path.join(seq_path, f'{chr1_name}.fa.gz')
    seq_file_path2 = os.path.join(seq_path, f'{chr2_name}.fa.gz')
    print(f"使用序列文件: {seq_file_path1}, {seq_file_path2}")
    
    # Load first chromosome region (from start1 to pos1, 1MB length)
    seq_region1, chip_region1 = infer.load_region(chr1_name, start1, seq_file_path1, chip_path, window=1048576)
    
    # Load second chromosome region (from pos2 to pos2+1MB, 1MB length)  
    seq_region2, chip_region2 = infer.load_region(chr2_name, pos2, seq_file_path2, chip_path, window=1048576)
    
    # Load full 2MB regions for individual predictions
    start1_full = max(0, pos1 - 1048576)
    start2_full = max(0, pos2 - 1048576) 
    
    print(f"Loading full Chr1 region: {chr1_name}:{start1_full}-{start1_full + 2097152}")
    print(f"Loading full Chr2 region: {chr2_name}:{start2_full}-{start2_full + 2097152}")
    
    seq_region1_full, chip_region1_full = infer.load_region(chr1_name, start1_full, seq_file_path1, chip_path)
    seq_region2_full, chip_region2_full = infer.load_region(chr2_name, start2_full, seq_file_path2, chip_path)
    
    # Load ground truth Hi-C data
    print("Loading ground truth Hi-C data...")
    hic_feature1 = data_feature.HiCFeature(path=f'{hic_path}/{chr1_name}.npz')
    hic_feature2 = data_feature.HiCFeature(path=f'{hic_path}/{chr2_name}.npz')
    
    # Get ground truth Hi-C matrices
    hic_gt_chr1 = hic_feature1.get(start1_full)
    hic_gt_chr2 = hic_feature2.get(start2_full)
    
    print(f"Ground truth Hi-C matrix shapes: Chr1={hic_gt_chr1.shape}, Chr2={hic_gt_chr2.shape}")
    
    # Process ground truth data first to normalize to 256x256
    hic_gt_chr1 = resize(hic_gt_chr1, (256, 256), anti_aliasing=True)
    hic_gt_chr1 = np.log(hic_gt_chr1 + 1)
    
    hic_gt_chr2 = resize(hic_gt_chr2, (256, 256), anti_aliasing=True)
    hic_gt_chr2 = np.log(hic_gt_chr2 + 1)
    
    # Create ground truth translocation by taking only quadrants II and IV
    # 只保留二四象限：Chr1的左上部分 + Chr2的右下部分
    half_size = 128  # 256/2 = 128
    
    # Create the fused ground truth matrix
    hic_gt_translocated = np.zeros((256, 256))
    hic_gt_translocated[:half_size, :half_size] = hic_gt_chr1[:half_size, :half_size]  # Chr1的左上象限（第二象限）
    hic_gt_translocated[half_size:, half_size:] = hic_gt_chr2[half_size:, half_size:]  # Chr2的右下象限（第四象限）
    
    # Create translocated version by concatenating the two halves
    seq_region_translocated = np.concatenate([seq_region1, seq_region2], axis=0)
    chip_region_translocated = np.concatenate([chip_region1, chip_region2], axis=0)
    
    print(f"Translocated sequence shape: {seq_region_translocated.shape}")
    print(f"Translocated CHIP shape: {chip_region_translocated.shape}")
    
    # Predictions
    print("Predicting original Chr1 region...")
    pred_chr1 = infer.prediction(seq_region1_full, chip_region1_full, model_path)
    
    print("Predicting original Chr2 region...")
    pred_chr2 = infer.prediction(seq_region2_full, chip_region2_full, model_path)
    
    print("Predicting translocated region...")
    pred_translocated = infer.prediction(seq_region_translocated, chip_region_translocated, model_path)
    
    # Prepare 4th and 5th panel data based on analysis mode
    if use_prediction_diff:
        print("Using prediction difference mode...")
        # 4th panel: concatenated predictions (只保留二四象限)
        half_size = 128
        pred_concatenated = np.zeros((256, 256))
        pred_concatenated[:half_size, :half_size] = pred_chr1[:half_size, :half_size]  # Chr1 的左上象限（第二象限）
        pred_concatenated[half_size:, half_size:] = pred_chr2[half_size:, half_size:]  # Chr2 的右下象限（第四象限）
        
        # 5th panel: difference between concatenated predictions and translocation prediction
        diff_pred_pred = pred_concatenated - pred_translocated
        
        fourth_panel = pred_concatenated
        fifth_panel = diff_pred_pred
        fourth_panel_title = 'Concatenated Predictions'
        fifth_panel_title = 'Difference (Concat - Trans)'
        analysis_mode = 'prediction_diff'
        
        print(f"Concatenated prediction shape: {pred_concatenated.shape}")
        print(f"Prediction difference shape: {diff_pred_pred.shape}")
        
    else:
        print("Using ground truth mode...")
        # 4th panel: ground truth
        # 5th panel: difference between translocation prediction and ground truth
        diff_pred_gt = pred_translocated - hic_gt_translocated
        
        fourth_panel = hic_gt_translocated
        fifth_panel = diff_pred_gt
        fourth_panel_title = 'Ground Truth Hi-C'
        fifth_panel_title = 'Difference (Prediction - GT)'
        analysis_mode = 'ground_truth'
    
    # Create comparison plot
    plot = MatrixPlotTranslocation(
        output_path, pred_chr1, pred_chr2, pred_translocated, hic_gt_chr1, hic_gt_chr2, 
        fourth_panel, fifth_panel, 'translocation',
        celltype, chr1_name, pos1, chr2_name, pos2,
        show_translocation_line=show_translocation_line,
        chip_chr1=chip_region1_full, chip_chr2=chip_region2_full, chip_translocated=chip_region_translocated,
        start1_full=start1_full, start2_full=start2_full,
        fourth_panel_title=fourth_panel_title, fifth_panel_title=fifth_panel_title,
        analysis_mode=analysis_mode)
    plot.plot()

class MatrixPlotTranslocation:
    """
    Custom plotting class for translocation comparison visualization
    Based on duplication visualization logic
    """
    
    def __init__(self, output_path, pred_chr1, pred_chr2, pred_translocated, 
                 hic_gt_chr1, hic_gt_chr2, fourth_panel, fifth_panel, prefix, 
                 celltype, chr1_name, pos1, chr2_name, pos2,
                 show_translocation_line=True, chip_chr1=None, chip_chr2=None, chip_translocated=None,
                 start1_full=None, start2_full=None, fourth_panel_title='Ground Truth Hi-C', 
                 fifth_panel_title='Difference (GT - Prediction)', analysis_mode='ground_truth'):
        self.output_path = output_path
        self.prefix = prefix
        self.celltype = celltype
        self.chr1_name = chr1_name
        self.pos1 = pos1
        self.chr2_name = chr2_name
        self.pos2 = pos2
        self.pred_chr1 = pred_chr1
        self.pred_chr2 = pred_chr2
        self.pred_translocated = pred_translocated
        self.hic_gt_chr1 = hic_gt_chr1
        self.hic_gt_chr2 = hic_gt_chr2
        self.fourth_panel = fourth_panel
        self.fifth_panel = fifth_panel
        self.fourth_panel_title = fourth_panel_title
        self.fifth_panel_title = fifth_panel_title
        self.analysis_mode = analysis_mode
        self.show_translocation_line = show_translocation_line
        self.chip_chr1 = chip_chr1
        self.chip_chr2 = chip_chr2
        self.chip_translocated = chip_translocated
        self.start1_full = start1_full
        self.start2_full = start2_full
        
        self.create_save_path(output_path, celltype, prefix)

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok=True)
        os.makedirs(f'{self.save_path}/npy', exist_ok=True)

    def get_colormap_hic(self):
        """Get colormap for Hi-C visualization (same as duplication.py)"""
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        return color_map
    
    def get_colormap_diff(self):
        """Get colormap for difference visualization"""
        return matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    
    def add_gray_overlay_difference(self, ax):
        """
        Add gray overlay rectangles for quadrants II and IV in difference plot
        灰显示二四象限，突出显示一三象限
        """
        from matplotlib.patches import Rectangle
        
        half_size = 128  # 256/2 = 128
        
        # Gray color with transparency
        gray_color = 'gray'
        alpha = 0.6  # Semi-transparent
        
        # 在imshow中，Rectangle(x, y, width, height)的坐标系：
        # x = 列坐标，y = 行坐标，(0,0)在左上角
        
        # 第二象限（左上）: 行[0:128], 列[0:128]
        quadrant_ii_rect = Rectangle((0, 0), half_size, half_size,
                                   facecolor=gray_color, alpha=alpha, edgecolor='none')
        ax.add_patch(quadrant_ii_rect)
        
        # 第四象限（右下）: 行[128:256], 列[128:256]
        quadrant_iv_rect = Rectangle((half_size, half_size), half_size, half_size,
                                   facecolor=gray_color, alpha=alpha, edgecolor='none')
        ax.add_patch(quadrant_iv_rect)
        
        print(f"Added gray overlay for quadrants II and IV in difference plot")
        print(f"  Quadrant II (left-top): 行[0:{half_size}], 列[0:{half_size}]")
        print(f"  Quadrant IV (right-bottom): 行[{half_size}:256], 列[{half_size}:256]")
    
    def extract_virtual_4c(self, matrix, viewpoint_bin):
        """
        Extract Virtual 4C profile from Hi-C matrix at a specific viewpoint
        
        Parameters:
        - matrix: Hi-C matrix (256x256)
        - viewpoint_bin: the bin index to use as viewpoint
        
        Returns:
        - virtual_4c: 1D array representing interactions from the viewpoint
        """
        # Ensure viewpoint is within bounds
        viewpoint_bin = max(0, min(viewpoint_bin, matrix.shape[0] - 1))
        
        # Extract the row at the viewpoint (interactions from this position to all others)
        virtual_4c = matrix[viewpoint_bin, :]
        
        return virtual_4c
    
    def plot_virtual_4c_tracks(self, axs):
        """Plot Virtual 4C tracks below Hi-C maps at translocation breakpoints"""
        # Breakpoint is at the center for translocated matrices (bin 128)
        breakpoint_bin = 128
        
        # For Chr1 and Chr2, breakpoint is at their respective positions
        scaling_factor = 256 / 2097152
        breakpoint1_relative = self.pos1 - self.start1_full
        breakpoint1_bin = int(breakpoint1_relative * scaling_factor)
        breakpoint1_bin = max(0, min(breakpoint1_bin, 255))
        
        breakpoint2_relative = self.pos2 - self.start2_full
        breakpoint2_bin = int(breakpoint2_relative * scaling_factor)
        breakpoint2_bin = max(0, min(breakpoint2_bin, 255))
        
        # Extract Virtual 4C profiles
        v4c_chr1 = self.extract_virtual_4c(self.pred_chr1, breakpoint1_bin)
        v4c_chr2 = self.extract_virtual_4c(self.pred_chr2, breakpoint2_bin)
        v4c_translocated = self.extract_virtual_4c(self.pred_translocated, breakpoint_bin)
        v4c_fourth = self.extract_virtual_4c(self.fourth_panel, breakpoint_bin)
        v4c_diff = v4c_translocated - v4c_fourth
        
        # Create genomic coordinates for x-axis (matching heatmap)
        window_size = 2097152
        
        # Chr1 and Chr2 positions in Mb
        positions_chr1 = np.linspace(self.start1_full, self.start1_full + window_size, 256) / 1e6
        positions_chr2 = np.linspace(self.start2_full, self.start2_full + window_size, 256) / 1e6
        
        # Translocated positions (0-2Mb, with breakpoint at 1Mb)
        positions_trans = np.linspace(0, window_size, 256) / 1e6
        
        viewpoint1_mb = self.pos1 / 1e6
        viewpoint2_mb = self.pos2 / 1e6
        viewpoint_trans_mb = 1.0  # 1 Mb = center
        
        # Plot Chr1 Virtual 4C
        axs["Chr1_Track"].fill_between(positions_chr1, 0, v4c_chr1, alpha=0.6, color='steelblue')
        axs["Chr1_Track"].plot(positions_chr1, v4c_chr1, 'b-', linewidth=0.8)
        axs["Chr1_Track"].axvline(x=viewpoint1_mb, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        axs["Chr1_Track"].set_ylabel('V4C')
        axs["Chr1_Track"].set_xlabel(f'{self.chr1_name} (Mb)')
        axs["Chr1_Track"].set_xlim(positions_chr1[0], positions_chr1[-1])
        axs["Chr1_Track"].set_title(f'V4C @ {viewpoint1_mb:.2f} Mb')
        
        # Plot Chr2 Virtual 4C
        axs["Chr2_Track"].fill_between(positions_chr2, 0, v4c_chr2, alpha=0.6, color='salmon')
        axs["Chr2_Track"].plot(positions_chr2, v4c_chr2, 'r-', linewidth=0.8)
        axs["Chr2_Track"].axvline(x=viewpoint2_mb, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
        axs["Chr2_Track"].set_ylabel('V4C')
        axs["Chr2_Track"].set_xlabel(f'{self.chr2_name} (Mb)')
        axs["Chr2_Track"].set_xlim(positions_chr2[0], positions_chr2[-1])
        axs["Chr2_Track"].set_title(f'V4C @ {viewpoint2_mb:.2f} Mb')
        
        # Plot Translocated Virtual 4C
        axs["Translocated_Track"].fill_between(positions_trans, 0, v4c_translocated, alpha=0.6, color='forestgreen')
        axs["Translocated_Track"].plot(positions_trans, v4c_translocated, 'g-', linewidth=0.8)
        axs["Translocated_Track"].axvline(x=viewpoint_trans_mb, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Breakpoint')
        axs["Translocated_Track"].set_ylabel('V4C')
        axs["Translocated_Track"].set_xlabel('Fusion Position (Mb)')
        axs["Translocated_Track"].set_xlim(positions_trans[0], positions_trans[-1])
        axs["Translocated_Track"].set_title(f'V4C @ Breakpoint')
        
        # Plot Fourth Panel Virtual 4C
        axs["GroundTruth_Track"].fill_between(positions_trans, 0, v4c_fourth, alpha=0.6, color='purple')
        axs["GroundTruth_Track"].plot(positions_trans, v4c_fourth, color='purple', linewidth=0.8)
        axs["GroundTruth_Track"].axvline(x=viewpoint_trans_mb, color='black', linestyle='--', linewidth=2, alpha=0.8)
        axs["GroundTruth_Track"].set_ylabel('V4C')
        axs["GroundTruth_Track"].set_xlabel('Fusion Position (Mb)')
        axs["GroundTruth_Track"].set_xlim(positions_trans[0], positions_trans[-1])
        axs["GroundTruth_Track"].set_title(f'V4C ({self.fourth_panel_title})')
        
        # Plot Difference Virtual 4C
        axs["Difference_Track"].fill_between(positions_trans, 0, v4c_diff, 
                                             where=(v4c_diff >= 0), alpha=0.6, color='red', label='Trans > Ref')
        axs["Difference_Track"].fill_between(positions_trans, 0, v4c_diff, 
                                             where=(v4c_diff < 0), alpha=0.6, color='blue', label='Trans < Ref')
        axs["Difference_Track"].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axs["Difference_Track"].axvline(x=viewpoint_trans_mb, color='black', linestyle='--', linewidth=2, alpha=0.8)
        axs["Difference_Track"].set_ylabel('Δ V4C')
        axs["Difference_Track"].set_xlabel('Fusion Position (Mb)')
        axs["Difference_Track"].set_xlim(positions_trans[0], positions_trans[-1])
        axs["Difference_Track"].set_title('V4C Difference')
    
    def plot_chipseq_tracks(self, axs):
        """Plot ChIP-seq tracks below Hi-C maps"""
        if self.chip_chr1 is None or self.chip_chr2 is None or self.chip_translocated is None:
            print("Warning: No ChIP-seq data available for track visualization")
            return
        
        # Create genomic coordinates for x-axis
        window_size = 2097152
        
        # Chr1 coordinates
        genomic_positions_chr1 = np.linspace(self.start1_full, self.start1_full + window_size, len(self.chip_chr1))
        
        # Chr2 coordinates  
        genomic_positions_chr2 = np.linspace(self.start2_full, self.start2_full + window_size, len(self.chip_chr2))
        
        # Translocated coordinates (split into two halves)
        half_size = len(self.chip_translocated) // 2
        genomic_positions_trans_1 = np.linspace(0, 1048576, half_size)  # First half (from Chr1)
        genomic_positions_trans_2 = np.linspace(1048576, 2097152, half_size)  # Second half (from Chr2)
        genomic_positions_trans = np.concatenate([genomic_positions_trans_1, genomic_positions_trans_2])
        
        # Plot Chr1 ChIP-seq track
        axs["Chr1_Track"].plot(genomic_positions_chr1, self.chip_chr1, 'b-', linewidth=1)
        axs["Chr1_Track"].set_title(f'{self.chr1_name} ChIP-seq')
        axs["Chr1_Track"].set_ylabel('Signal')
        axs["Chr1_Track"].set_xlabel(f'{self.chr1_name}:{self.start1_full}-{self.start1_full + window_size}')
        axs["Chr1_Track"].grid(True, alpha=0.3)
        
        # Plot Chr2 ChIP-seq track
        axs["Chr2_Track"].plot(genomic_positions_chr2, self.chip_chr2, 'r-', linewidth=1)
        axs["Chr2_Track"].set_title(f'{self.chr2_name} ChIP-seq')
        axs["Chr2_Track"].set_ylabel('Signal')
        axs["Chr2_Track"].set_xlabel(f'{self.chr2_name}:{self.start2_full}-{self.start2_full + window_size}')
        axs["Chr2_Track"].grid(True, alpha=0.3)
        
        # Plot translocated ChIP-seq track
        axs["Translocated_Track"].plot(genomic_positions_trans, self.chip_translocated, 'g-', linewidth=1)
        axs["Translocated_Track"].axvline(x=1048576, color='k', linestyle='--', alpha=0.7, label='Breakpoint')
        axs["Translocated_Track"].set_title(f'Translocated ChIP-seq ({self.chr1_name}+{self.chr2_name})')
        axs["Translocated_Track"].set_ylabel('Signal')
        axs["Translocated_Track"].set_xlabel('Position (bp) - Chr1|Chr2 fusion')
        axs["Translocated_Track"].grid(True, alpha=0.3)
        
        # Plot ground truth ChIP-seq track (same as translocated)
        axs["GroundTruth_Track"].plot(genomic_positions_trans, self.chip_translocated, 'purple', linewidth=1)
        axs["GroundTruth_Track"].axvline(x=1048576, color='k', linestyle='--', alpha=0.7, label='Breakpoint')
        axs["GroundTruth_Track"].set_title(f'Ground Truth ChIP-seq ({self.chr1_name}+{self.chr2_name})')
        axs["GroundTruth_Track"].set_ylabel('Signal')
        axs["GroundTruth_Track"].set_xlabel('Position (bp) - Chr1|Chr2 fusion')
        axs["GroundTruth_Track"].grid(True, alpha=0.3)
        
        # Plot difference ChIP-seq track (same as translocated, since we use same data)
        axs["Difference_Track"].plot(genomic_positions_trans, self.chip_translocated, 'orange', linewidth=1)
        axs["Difference_Track"].axvline(x=1048576, color='k', linestyle='--', alpha=0.7, label='Breakpoint')
        axs["Difference_Track"].set_title(f'Difference ChIP-seq ({self.chr1_name}+{self.chr2_name})')
        axs["Difference_Track"].set_ylabel('Signal')
        axs["Difference_Track"].set_xlabel('Position (bp) - Chr1|Chr2 fusion')
        axs["Difference_Track"].grid(True, alpha=0.3)
        
        # Add breakpoint highlighting with consistent width
        if self.show_translocation_line:
            # Use consistent breakpoint width (1000bp on each side)
            breakpoint_width = 1000
            
            # Highlight breakpoint regions
            axs["Chr1_Track"].axvspan(self.pos1 - breakpoint_width, self.pos1 + breakpoint_width, 
                                     alpha=0.2, color='red', label='Breakpoint region')
            axs["Chr2_Track"].axvspan(self.pos2 - breakpoint_width, self.pos2 + breakpoint_width, 
                                     alpha=0.2, color='red', label='Breakpoint region')
            
            # For translocated tracks, highlight at the center (1048576)
            axs["Translocated_Track"].axvspan(1048576 - breakpoint_width, 1048576 + breakpoint_width, 
                                             alpha=0.2, color='red', label='Breakpoint region')
            axs["GroundTruth_Track"].axvspan(1048576 - breakpoint_width, 1048576 + breakpoint_width, 
                                            alpha=0.2, color='red', label='Breakpoint region')
            axs["Difference_Track"].axvspan(1048576 - breakpoint_width, 1048576 + breakpoint_width, 
                                           alpha=0.2, color='red', label='Breakpoint region')
            
            # Add legend
            axs["Chr1_Track"].legend(loc='upper right')
            axs["Chr2_Track"].legend(loc='upper right')
            axs["Translocated_Track"].legend(loc='upper right')

    def plot(self):
        """Create the five-panel comparison plot with Virtual 4C tracks"""
        plt.rcParams.update({'font.size': 12})
        
        # Create figure with 2 rows: Hi-C maps on top, Virtual 4C tracks on bottom
        fig = plt.figure(figsize=(40, 18), constrained_layout=True)
        axs = fig.subplot_mosaic([
            ['Chr1_HiC', 'Chr2_HiC', 'Translocated_HiC', 'GroundTruth_HiC', 'Difference_HiC'],
            ['Chr1_Track', 'Chr2_Track', 'Translocated_Track', 'GroundTruth_Track', 'Difference_Track']
        ], height_ratios=[3, 1])
        
        fig.suptitle(f'Translocation simulation: {self.chr1_name}:{self.pos1} ↔ {self.chr2_name}:{self.pos2}')

        # Panel 1: Chr1 Hi-C prediction
        color_map_hic = self.get_colormap_hic()
        axs["Chr1_HiC"].set_title(f'{self.chr1_name} Hi-C prediction')
        axs["Chr1_HiC"].imshow(self.pred_chr1, cmap=color_map_hic, vmin=0, vmax=3)
        axs["Chr1_HiC"].get_xaxis().set_ticks([])
        axs["Chr1_HiC"].get_yaxis().set_visible(False)

        # Panel 2: Chr2 Hi-C prediction
        axs["Chr2_HiC"].set_title(f'{self.chr2_name} Hi-C prediction')
        axs["Chr2_HiC"].imshow(self.pred_chr2, cmap=color_map_hic, vmin=0, vmax=3)
        axs["Chr2_HiC"].get_xaxis().set_ticks([])
        axs["Chr2_HiC"].get_yaxis().set_visible(False)

        # Panel 3: Translocated Hi-C prediction
        axs["Translocated_HiC"].set_title(f'Translocated Hi-C prediction ({self.chr1_name}+{self.chr2_name})')
        axs["Translocated_HiC"].imshow(self.pred_translocated, cmap=color_map_hic, vmin=0, vmax=3)
        axs["Translocated_HiC"].get_xaxis().set_ticks([])
        axs["Translocated_HiC"].get_yaxis().set_visible(False)

        # Panel 4: Fourth panel (either Ground Truth or Concatenated Predictions)
        if self.analysis_mode == 'prediction_diff':
            axs["GroundTruth_HiC"].set_title(f'{self.fourth_panel_title} ({self.chr1_name}+{self.chr2_name})')
            axs["GroundTruth_HiC"].imshow(self.fourth_panel, cmap=color_map_hic, vmin=0, vmax=3)
        else:
            axs["GroundTruth_HiC"].set_title(f'{self.fourth_panel_title} ({self.chr1_name}+{self.chr2_name})')
            axs["GroundTruth_HiC"].imshow(self.fourth_panel, cmap=color_map_hic, vmin=0, vmax=3)
        axs["GroundTruth_HiC"].get_xaxis().set_ticks([])
        axs["GroundTruth_HiC"].get_yaxis().set_visible(False)

        # Panel 5: Fifth panel (either GT-Prediction difference or Prediction-Prediction difference)
        color_map_diff = self.get_colormap_diff()
        axs["Difference_HiC"].set_title(self.fifth_panel_title)
        axs["Difference_HiC"].imshow(self.fifth_panel, cmap=color_map_diff, vmin=-2, vmax=2)
        axs["Difference_HiC"].get_xaxis().set_ticks([])
        axs["Difference_HiC"].get_yaxis().set_visible(False)
        
        # Add gray overlay to highlight quadrants I and III (by graying out II and IV)
        self.add_gray_overlay_difference(axs["Difference_HiC"])

        # Virtual 4C tracks (replacing ChIP-seq tracks)
        self.plot_virtual_4c_tracks(axs)

        # Add translocation breakpoint lines to Hi-C maps if requested
        if self.show_translocation_line:
            DASH_W = 3.0
            DASHES = (4, 4)
            
            # Convert genomic positions to pixel coordinates
            # Assuming 256 pixels represents 2097152 bp
            scaling_factor = 256 / 2097152
            
            # For Chr1: breakpoint is at pos1, relative to start1_full
            breakpoint1_relative = self.pos1 - self.start1_full
            breakpoint1_px = breakpoint1_relative * scaling_factor
            
            # For Chr2: breakpoint is at pos2, relative to start2_full
            breakpoint2_relative = self.pos2 - self.start2_full
            breakpoint2_px = breakpoint2_relative * scaling_factor
            
            # For translocated: breakpoint is at the middle (128px)
            breakpoint_trans_px = 128  # Middle of 256px
            
            # Add lines to Hi-C panels
            if 0 <= breakpoint1_px <= 256:
                axs["Chr1_HiC"].axvline(x=breakpoint1_px, color='r', ls="--", dashes=DASHES, lw=DASH_W)
                axs["Chr1_HiC"].axhline(y=breakpoint1_px, color='r', ls="--", dashes=DASHES, lw=DASH_W)
            
            if 0 <= breakpoint2_px <= 256:
                axs["Chr2_HiC"].axvline(x=breakpoint2_px, color='r', ls="--", dashes=DASHES, lw=DASH_W)
                axs["Chr2_HiC"].axhline(y=breakpoint2_px, color='r', ls="--", dashes=DASHES, lw=DASH_W)
            
            # Translocation breakpoint at the center for all translocated panels
            for panel in ["Translocated_HiC", "GroundTruth_HiC", "Difference_HiC"]:
                axs[panel].axvline(x=breakpoint_trans_px, color='k', ls="--", dashes=DASHES, lw=DASH_W)
                axs[panel].axhline(y=breakpoint_trans_px, color='k', ls="--", dashes=DASHES, lw=DASH_W)

        self.save_data(plt)

    def save_data(self, plt):
        """Save the plot and numerical data"""
        mode_suffix = f'_{self.analysis_mode}' if self.analysis_mode != 'ground_truth' else ''
        filename_base = f'{self.chr1_name}_{self.pos1}_{self.chr2_name}_{self.pos2}_translocation{mode_suffix}'
        
        # Save plot
        plt.savefig(f'{self.save_path}/imgs/{filename_base}.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # Save numerical data - basic predictions always saved
        np.save(f'{self.save_path}/npy/{filename_base}_chr1_pred', self.pred_chr1)
        np.save(f'{self.save_path}/npy/{filename_base}_chr2_pred', self.pred_chr2)
        np.save(f'{self.save_path}/npy/{filename_base}_translocated_pred', self.pred_translocated)
        np.save(f'{self.save_path}/npy/{filename_base}_chr1_gt', self.hic_gt_chr1)
        np.save(f'{self.save_path}/npy/{filename_base}_chr2_gt', self.hic_gt_chr2)
        
        # Save mode-specific data
        if self.analysis_mode == 'prediction_diff':
            np.save(f'{self.save_path}/npy/{filename_base}_concatenated_pred', self.fourth_panel)
            np.save(f'{self.save_path}/npy/{filename_base}_pred_diff', self.fifth_panel)
            file_count = 7
            specific_files = "concatenated_pred, pred_diff"
        else:  # ground_truth mode
            np.save(f'{self.save_path}/npy/{filename_base}_translocated_gt', self.fourth_panel)
            np.save(f'{self.save_path}/npy/{filename_base}_pred_gt_diff', self.fifth_panel)
            file_count = 7
            specific_files = "translocated_gt, pred_gt_diff"
        
        print(f"Results saved to {self.save_path}")
        print(f"  - Images: {filename_base}.png")
        print(f"  - Arrays: {filename_base}_*.npy ({file_count} files)")
        print(f"  - Mode: {self.analysis_mode} ({specific_files})")

if __name__ == '__main__':
    main() 
