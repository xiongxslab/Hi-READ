import os
import numpy as np
import pandas as pd
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import plot_utils

def main():
    parser = argparse.ArgumentParser(description='Hi-READ Duplication Module.')
    
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

    # Duplication related params
    parser.add_argument('--dup-source-start', dest='dup_source_start', type=int,
                        help='Starting point of the source region for duplication (relative to start position).', required=True)
    parser.add_argument('--dup-source-width', dest='dup_source_width', type=int,
                        help='Width of the source region for duplication.', required=True)
    parser.add_argument('--dup-target-start', dest='dup_target_start', type=int,
                        help='Starting point of the target region for duplication (relative to start position).', required=True)
    parser.add_argument('--chip-mode', dest='chip_mode', 
                        default='copy',
                        choices=['copy', 'zero', 'mean'],
                        help='ChIP-seq processing mode: copy (copy corresponding signals), zero (set to 0), mean (set to matrix mean) (default: %(default)s)')
    parser.add_argument('--hide-line', dest='hide_duplication_line', 
                        action='store_true',
                        help='Remove the line showing duplication sites (default: %(default)s)')

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    single_duplication(args.output_path, args.celltype, args.chr_name, args.start, 
                      args.dup_source_start, args.dup_source_width, args.dup_target_start,
                      args.model_path, args.seq_path, args.chip_path, 
                      chip_mode=args.chip_mode,
                      show_duplication_line=not args.hide_duplication_line)

def calculate_difference_modified(hic_original, hic_duplication, dup_source_start, dup_source_width):
    """
    Calculate difference matrix with modified method for specific region
    
    Parameters:
    - hic_original: original Hi-C matrix (256x256)
    - hic_duplication: duplicated Hi-C matrix (256x256) 
    - dup_source_start: starting position of duplication source (in matrix coordinates)
    - dup_source_width: width of duplication source (in matrix coordinates)
    
    Returns:
    - difference_matrix: modified difference matrix
    """
    # Convert genomic coordinates to matrix coordinates
    # Assuming 256 pixels represents 2097152 bp
    scaling_factor = 256 / 2097152
    i = int(dup_source_start * scaling_factor)  # source start in matrix coordinates
    L = int(dup_source_width * scaling_factor)  # source width in matrix coordinates
    
    # Initialize difference matrix with standard calculation
    difference_matrix = hic_duplication - hic_original
    
    # Ensure indices are within bounds
    if i + 2*L < 256 and i + L < 256 - L:
        # Modified calculation for specific region [i+2L:256, i+2L:256]
        # Use hic_duplication[i+2L:256,i+2L:256] - hic_original[i+L:256-L,i+L:256-L]
        
        # Extract the regions
        dup_region = hic_duplication[i+2*L:256, i+2*L:256]
        orig_region = hic_original[i+L:256-L, i+L:256-L]
        
        # Verify shapes match
        print(f"Shape verification:")
        print(f"  dup_region shape: {dup_region.shape}")
        print(f"  orig_region shape: {orig_region.shape}")
        print(f"  i={i}, L={L}, matrix coordinates")
        
        if dup_region.shape == orig_region.shape:
            # Apply modified calculation to the specific region
            difference_matrix[i+2*L:256, i+2*L:256] = dup_region - orig_region
            print(f"Modified difference calculation applied to region [{i+2*L}:256, {i+2*L}:256]")
        else:
            print(f"Warning: Shape mismatch, using standard calculation")
    else:
        print(f"Warning: Indices out of bounds (i={i}, L={L}, i+2L={i+2*L}), using standard calculation")
    
    return difference_matrix

def single_duplication(output_path, celltype, chr_name, start, dup_source_start, dup_source_width, 
                      dup_target_start, model_path, seq_path, chip_path, 
                      chip_mode='copy', show_duplication_line=True):
    """
    Perform single tandem duplication prediction and visualization
    
    Parameters:
    - dup_source_start: relative position of source region start (from prediction start)
    - dup_source_width: width of the region to be duplicated
    - dup_target_start: (legacy parameter, not used in tandem mode)
    - chip_mode: 'copy' (copy corresponding signals), 'zero' (set to 0), 'mean' (set to matrix mean)
    
    Tandem duplication mode: source region is duplicated and inserted right after itself
    """
    
    # Build full sequence file path
    seq_file_path = os.path.join(seq_path, f'{chr_name}.fa.gz')
    print(f"使用序列文件: {seq_file_path}")
    
    # Load original region
    seq_region, chip_region = infer.load_region(chr_name, start, seq_file_path, chip_path)
    
    # Create duplicated version (target_start is not used in tandem mode)
    seq_region_dup, chip_region_dup = duplication_with_modes(
        seq_region, chip_region, dup_source_start, dup_source_width, None, chip_mode)
    
    # Predictions
    print("Predicting original region...")
    pred_original = infer.prediction(seq_region, chip_region, model_path)
    
    print("Predicting tandem duplicated region...")
    pred_duplicated = infer.prediction(seq_region_dup, chip_region_dup, model_path)
    
    # Calculate difference with modified method
    diff_map = calculate_difference_modified(pred_original, pred_duplicated, dup_source_start, dup_source_width)
    
    # Create comparison plot
    plot = MatrixPlotDuplication(
        output_path, pred_original, pred_duplicated, diff_map, 'tandem_duplication',
        celltype, chr_name, start, dup_source_start, dup_source_width, dup_target_start,
        chip_mode=chip_mode, show_duplication_line=show_duplication_line,
        chip_original=chip_region, chip_duplicated=chip_region_dup)
    plot.plot()

def duplication_with_modes(seq_region, chip_region, source_start, source_width, target_start, chip_mode):
    # Note: target_start parameter is kept for backward compatibility but not used in tandem mode
    """
    Apply tandem duplication: insert duplicated region after source region, shifting downstream sequences
    
    New mode: A region is duplicated and inserted right after itself, pushing downstream content
    [A][B][C][D] -> [A][A'][B][C] (D is pushed out of the window)
    """
    # Copy the arrays to avoid modifying originals
    seq_dup = seq_region.copy()
    chip_dup = chip_region.copy()
    
    source_end = source_start + source_width
    window_size = len(seq_region)  # Usually 2097152 / 512 = 4096 for sequence
    
    # Ensure source region is valid
    if source_end > window_size:
        raise ValueError(f"Source region ({source_start}-{source_end}) goes beyond sequence boundaries ({window_size})")
    
    # Calculate insertion point (right after source region)
    insert_pos = source_end
    
    # Extract the source region to be duplicated
    source_seq = seq_region[source_start:source_end]
    source_chip = chip_region[source_start:source_end]
    
    # Create new sequences with tandem duplication
    # Part 1: Everything before and including source region
    seq_part1 = seq_dup[:insert_pos]
    chip_part1 = chip_dup[:insert_pos]
    
    # Part 2: Duplicated source region
    seq_part2 = source_seq
    if chip_mode == 'copy':
        chip_part2 = source_chip
    elif chip_mode == 'zero':
        chip_part2 = np.zeros_like(source_chip)
    elif chip_mode == 'mean':
        mean_value = np.nanmean(chip_region)
        chip_part2 = np.full_like(source_chip, mean_value)
    else:
        raise ValueError(f"Unknown chip_mode: {chip_mode}")
    
    # Part 3: Downstream sequence (shifted, truncated if necessary)
    remaining_space = window_size - len(seq_part1) - len(seq_part2)
    if remaining_space > 0:
        downstream_length = min(remaining_space, len(seq_dup) - insert_pos)
        seq_part3 = seq_dup[insert_pos:insert_pos + downstream_length]
        chip_part3 = chip_dup[insert_pos:insert_pos + downstream_length]
    else:
        # No space for downstream sequence
        seq_part3 = np.array([])
        chip_part3 = np.array([])
    
    # Concatenate all parts
    seq_dup_new = np.concatenate([seq_part1, seq_part2, seq_part3])
    chip_dup_new = np.concatenate([chip_part1, chip_part2, chip_part3])
    
    # Ensure we maintain the original window size
    if len(seq_dup_new) > window_size:
        seq_dup_new = seq_dup_new[:window_size]
        chip_dup_new = chip_dup_new[:window_size]
    elif len(seq_dup_new) < window_size:
        # Pad with zeros if needed (shouldn't happen in normal cases)
        pad_size = window_size - len(seq_dup_new)
        seq_dup_new = np.concatenate([seq_dup_new, np.zeros(pad_size)])
        chip_dup_new = np.concatenate([chip_dup_new, np.zeros(pad_size)])
    
    print(f"Tandem duplication: source {source_start}-{source_end} inserted at {insert_pos}")
    print(f"Window size: {window_size}, final size: {len(seq_dup_new)}")
    print(f"Downstream shift: {source_width} bp pushed downstream")
    
    return seq_dup_new, chip_dup_new

class MatrixPlotDuplication:
    """
    Custom plotting class for duplication comparison visualization
    Similar to the provided transduplication visualization
    """
    
    def __init__(self, output_path, pred_original, pred_duplicated, diff_map, prefix, 
                 celltype, chr_name, start_pos, dup_source_start, dup_source_width, 
                 dup_target_start, chip_mode='copy', show_duplication_line=True,
                 chip_original=None, chip_duplicated=None):
        self.output_path = output_path
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.pred_original = pred_original
        self.pred_duplicated = pred_duplicated
        self.diff_map = diff_map
        self.dup_source_start = dup_source_start
        self.dup_source_width = dup_source_width
        self.dup_target_start = dup_target_start
        self.chip_mode = chip_mode
        self.show_duplication_line = show_duplication_line
        self.chip_original = chip_original
        self.chip_duplicated = chip_duplicated
        
        self.create_save_path(output_path, celltype, prefix)

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok=True)
        os.makedirs(f'{self.save_path}/npy', exist_ok=True)

    # def get_colormap_hic(self):
    #     """Get colormap for Hi-C visualization (similar to original code)"""
    #     return matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    def get_colormap_hic(self):
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        return color_map
    def get_colormap_diff(self):
        """Get colormap for difference visualization"""
        return matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    
    def create_original_display_matrix(self):
        """
        Create a modified original matrix for visualization with special region highlighting
        """
        # Convert genomic coordinates to matrix coordinates
        scaling_factor = 256 / 2097152
        i = int(self.dup_source_start * scaling_factor)
        L = int(self.dup_source_width * scaling_factor)
        
        # Create a copy of the original matrix
        display_matrix = self.pred_original.copy()
        
        # Ensure indices are within bounds
        if i + L < 256 and 256 - L > i + L:
            # Convert the special region to grayscale
            # The special region is hic_original[i+L:256, i+L:256] minus hic_original[i+L:256-L, i+L:256-L]
            # This includes three parts:
            # 1. Right strip: [i+L:256-L, 256-L:256]
            # 2. Bottom strip: [256-L:256, i+L:256-L]  
            # 3. Bottom-right corner: [256-L:256, 256-L:256]
            
            # Use a very low value to make it appear dark/black in the red colormap
            # Since colormap goes from white(1,1,1) to red(1,0,0), low values appear dark
            gray_value = 0.2  # Very low value for dark gray/black appearance
            
            # Right strip
            if 256 - L < 256:
                display_matrix[i+L:256-L, 256-L:256] = gray_value
            
            # Bottom strip  
            if 256 - L < 256:
                display_matrix[256-L:256, i+L:256-L] = gray_value
            
            # Bottom-right corner
            if 256 - L < 256:
                display_matrix[256-L:256, 256-L:256] = gray_value
                
            print(f"Special region converted to grayscale: i={i}, L={L}")
            print(f"  Right strip: [{i+L}:{256-L}, {256-L}:256]")
            print(f"  Bottom strip: [{256-L}:256, {i+L}:{256-L}]")
            print(f"  Corner: [{256-L}:256, {256-L}:256]")
        
        return display_matrix
    
    def add_gray_overlay(self, ax):
        """
        Add gray overlay rectangles for the L-shaped region
        在imshow中，坐标是(x, y) = (列, 行)，原点在左上角
        """
        from matplotlib.patches import Rectangle
        
        # Convert genomic coordinates to matrix coordinates
        scaling_factor = 256 / 2097152
        i = int(self.dup_source_start * scaling_factor)
        L = int(self.dup_source_width * scaling_factor)
        
        # Ensure indices are within bounds
        if i + L < 256 and 256 - L > i + L:
            # Gray color with transparency
            gray_color = 'gray'
            alpha = 0.8  # More opaque
            
            # 在imshow中，Rectangle(x, y, width, height)的坐标系：
            # x = 列坐标，y = 行坐标，(0,0)在左上角
            
            # Right strip: 行[i+L:256-L], 列[256-L:256] = [73:244, 244:256]
            right_rect = Rectangle((256-L, i+L), L, (256-L)-(i+L),
                                 facecolor=gray_color, alpha=alpha, edgecolor='none')
            ax.add_patch(right_rect)
            
            # Bottom strip: 行[256-L:256], 列[i+L:256-L] = [244:256, 73:244]  
            bottom_rect = Rectangle((i+L, 256-L), (256-L)-(i+L), L,
                                  facecolor=gray_color, alpha=alpha, edgecolor='none')
            ax.add_patch(bottom_rect)
            
            # Bottom-right corner: 行[256-L:256], 列[256-L:256] = [244:256, 244:256]
            corner_rect = Rectangle((256-L, 256-L), L, L,
                                  facecolor=gray_color, alpha=alpha, edgecolor='none')
            ax.add_patch(corner_rect)
                
            print(f"Added gray overlay rectangles: i={i}, L={L}")
            print(f"  i+L={i+L}, 256-L={256-L}")
            print(f"  Right strip: 行[{i+L}:{256-L}], 列[{256-L}:256]")
            print(f"  Bottom strip: 行[{256-L}:256], 列[{i+L}:{256-L}]")
            print(f"  Corner: 行[{256-L}:256], 列[{256-L}:256]")
    
    def add_grayscale_region_lines(self, ax):
        """
        Add black dashed lines around the grayscale region in the original Hi-C plot
        """
        # Convert genomic coordinates to matrix coordinates
        scaling_factor = 256 / 2097152
        i = int(self.dup_source_start * scaling_factor)
        L = int(self.dup_source_width * scaling_factor)
        
        # Ensure indices are within bounds
        if i + L < 256 and 256 - L > i + L:
            # Line style parameters - make more visible
            line_color = 'white'  # 改为白色，在红色热图上更明显
            line_width = 3        # 增加线宽
            line_style = '--'
            
            # The grayscale region forms an L-shape containing:
            # - Right strip: [i+L:256-L, 256-L:256]
            # - Bottom strip: [256-L:256, i+L:256-L]  
            # - Corner: [256-L:256, 256-L:256]
            
            # Draw the boundary of this L-shape
            # Note: in matplotlib plot, first arg is x (column), second is y (row)
            
            # Draw the complete L-shaped boundary
            # The boundary forms a closed L-shape around the grayscale region
            
            # 1. Top edge: horizontal line at y=i+L, from x=256-L to x=256
            ax.plot([256-L, 256], [i+L, i+L], color=line_color, linewidth=line_width, linestyle=line_style)
            
            # 2. Right edge: vertical line at x=256, from y=i+L to y=256
            ax.plot([256, 256], [i+L, 256], color=line_color, linewidth=line_width, linestyle=line_style)
            
            # 3. Bottom edge: horizontal line at y=256, from x=256 to x=i+L
            ax.plot([256, i+L], [256, 256], color=line_color, linewidth=line_width, linestyle=line_style)
            
            # 4. Left edge (bottom part): vertical line at x=i+L, from y=256 to y=256-L
            ax.plot([i+L, i+L], [256, 256-L], color=line_color, linewidth=line_width, linestyle=line_style)
            
            # 5. Inner horizontal edge: horizontal line at y=256-L, from x=i+L to x=256-L
            ax.plot([i+L, 256-L], [256-L, 256-L], color=line_color, linewidth=line_width, linestyle=line_style)
            
            # 6. Inner vertical edge: vertical line at x=256-L, from y=256-L to y=i+L
            ax.plot([256-L, 256-L], [256-L, i+L], color=line_color, linewidth=line_width, linestyle=line_style)
            
            print(f"Added grayscale region boundary lines:")
            print(f"  L-shape boundary: i={i}, L={L}, region=[{i+L}:{256-L},{256-L}:256] + [{256-L}:256,{i+L}:{256-L}]")
    
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
    
    def plot_virtual_4c_tracks(self, axs, source_start_genomic, source_end_genomic):
        """Plot Virtual 4C tracks below Hi-C maps at duplication boundaries"""
        # Calculate viewpoint bins (duplication start and end)
        scaling_factor = 256 / 2097152
        viewpoint_start_bin = int(self.dup_source_start * scaling_factor)
        viewpoint_end_bin = int((self.dup_source_start + self.dup_source_width) * scaling_factor)
        
        # Extract Virtual 4C profiles at duplication start position
        v4c_original = self.extract_virtual_4c(self.pred_original, viewpoint_start_bin)
        v4c_duplicated = self.extract_virtual_4c(self.pred_duplicated, viewpoint_start_bin)
        v4c_diff = v4c_duplicated - v4c_original
        
        # Create genomic coordinates for x-axis (matching heatmap)
        window_size = 2097152
        genomic_positions = np.linspace(self.start_pos, self.start_pos + window_size, 256) / 1e6  # Convert to Mb
        
        viewpoint_genomic = (self.start_pos + self.dup_source_start) / 1e6
        
        # Plot Original Virtual 4C
        axs["Original_Track"].fill_between(genomic_positions, 0, v4c_original, alpha=0.6, color='steelblue')
        axs["Original_Track"].plot(genomic_positions, v4c_original, 'b-', linewidth=0.8)
        axs["Original_Track"].set_ylabel('V4C Signal')
        axs["Original_Track"].set_xlim(genomic_positions[0], genomic_positions[-1])
        axs["Original_Track"].set_title(f'Virtual 4C (viewpoint: {viewpoint_genomic:.3f} Mb)')
        
        # Plot Duplicated Virtual 4C
        axs["Duplicated_Track"].fill_between(genomic_positions, 0, v4c_duplicated, alpha=0.6, color='salmon')
        axs["Duplicated_Track"].plot(genomic_positions, v4c_duplicated, 'r-', linewidth=0.8)
        axs["Duplicated_Track"].set_ylabel('V4C Signal')
        axs["Duplicated_Track"].set_xlim(genomic_positions[0], genomic_positions[-1])
        axs["Duplicated_Track"].set_title(f'Virtual 4C After Duplication')
        
        # Plot Difference Virtual 4C
        axs["Difference_Track"].fill_between(genomic_positions, 0, v4c_diff, 
                                             where=(v4c_diff >= 0), alpha=0.6, color='red', label='Increased')
        axs["Difference_Track"].fill_between(genomic_positions, 0, v4c_diff, 
                                             where=(v4c_diff < 0), alpha=0.6, color='blue', label='Decreased')
        axs["Difference_Track"].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axs["Difference_Track"].set_ylabel('Δ V4C')
        axs["Difference_Track"].set_xlim(genomic_positions[0], genomic_positions[-1])
        axs["Difference_Track"].set_title('Virtual 4C Difference')
        
        # Add viewpoint line to all tracks
        for track_name in ["Original_Track", "Duplicated_Track", "Difference_Track"]:
            axs[track_name].axvline(x=viewpoint_genomic, color='purple', linestyle='--', 
                                   linewidth=1.5, alpha=0.8, label='Viewpoint')
            axs[track_name].set_xlabel('Genomic Position (Mb)')
        
        # Add duplication region highlighting
        if self.show_duplication_line:
            source_start_mb = source_start_genomic / 1e6
            source_end_mb = source_end_genomic / 1e6
            insert_end_mb = (source_end_genomic + self.dup_source_width) / 1e6
            
            for track_name in ["Original_Track", "Duplicated_Track", "Difference_Track"]:
                axs[track_name].axvspan(source_start_mb, source_end_mb, 
                                       alpha=0.15, color='blue', label='Source region')
                
                if track_name in ["Duplicated_Track", "Difference_Track"]:
                    axs[track_name].axvspan(source_end_mb, insert_end_mb, 
                                           alpha=0.15, color='green', label='Inserted region')
    
    def plot_chipseq_tracks(self, axs, source_start_genomic, source_end_genomic):
        """Plot ChIP-seq tracks below Hi-C maps"""
        if self.chip_original is None or self.chip_duplicated is None:
            print("Warning: No ChIP-seq data available for track visualization")
            return
        
        # Create genomic coordinates for x-axis
        window_size = 2097152
        genomic_positions = np.linspace(self.start_pos, self.start_pos + window_size, len(self.chip_original))
        
        # Calculate difference track
        chip_diff = self.chip_duplicated - self.chip_original
        
        # Plot original ChIP-seq track
        axs["Original_Track"].plot(genomic_positions, self.chip_original, 'b-', linewidth=1)
        axs["Original_Track"].set_title('Original ChIP-seq')
        axs["Original_Track"].set_ylabel('Signal')
        axs["Original_Track"].set_xlabel(f'{self.chr_name}:{self.start_pos}-{self.start_pos + window_size}')
        axs["Original_Track"].grid(True, alpha=0.3)
        
        # Plot duplicated ChIP-seq track
        axs["Duplicated_Track"].plot(genomic_positions, self.chip_duplicated, 'r-', linewidth=1)
        axs["Duplicated_Track"].set_title(f'After duplication (mode: {self.chip_mode})')
        axs["Duplicated_Track"].set_ylabel('Signal')
        axs["Duplicated_Track"].set_xlabel(f'Source: {source_start_genomic}-{source_end_genomic}')
        axs["Duplicated_Track"].grid(True, alpha=0.3)
        
        # Plot difference track
        axs["Difference_Track"].plot(genomic_positions, chip_diff, 'g-', linewidth=1)
        axs["Difference_Track"].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axs["Difference_Track"].set_title('ChIP-seq difference (Dup - Orig)')
        axs["Difference_Track"].set_ylabel('Δ Signal')
        axs["Difference_Track"].set_xlabel('Genomic position')
        axs["Difference_Track"].grid(True, alpha=0.3)
        
        # Add source region highlighting to all tracks
        if self.show_duplication_line:
            for track_name in ["Original_Track", "Duplicated_Track", "Difference_Track"]:
                axs[track_name].axvspan(source_start_genomic, source_end_genomic, 
                                       alpha=0.2, color='blue', label='Source region')
                
                # For duplicated and difference tracks, also show the inserted region
                if track_name in ["Duplicated_Track", "Difference_Track"]:
                    insert_start = source_end_genomic
                    insert_end = insert_start + (source_end_genomic - source_start_genomic)
                    axs[track_name].axvspan(insert_start, insert_end, 
                                           alpha=0.2, color='green', label='Inserted region')
                    
                # Add legend only to the first track
                if track_name == "Original_Track":
                    axs[track_name].legend(loc='upper right')

    def plot(self):
        """Create the three-panel comparison plot with Virtual 4C tracks"""
        plt.rcParams.update({'font.size': 14})
        
        # Create figure with 2 rows: Hi-C maps on top, Virtual 4C tracks on bottom
        fig = plt.figure(figsize=(30, 18), constrained_layout=True)
        axs = fig.subplot_mosaic([
            ['Original_HiC', 'Duplicated_HiC', 'Difference_HiC'],
            ['Original_Track', 'Duplicated_Track', 'Difference_Track']
        ], height_ratios=[3, 1])
        
        # Calculate genomic coordinates for title
        source_start_genomic = self.start_pos + self.dup_source_start
        source_end_genomic = source_start_genomic + self.dup_source_width
        target_start_genomic = self.start_pos + self.dup_target_start
        target_end_genomic = target_start_genomic + self.dup_source_width
        
        fig.suptitle(f'Tandem duplication simulation of {self.chr_name}:{source_start_genomic}-{source_end_genomic} (inserted after source)')

        # Panel 1: Original Hi-C prediction with special region highlighting
        color_map_hic = self.get_colormap_hic()
        axs["Original_HiC"].set_title('Original Hi-C prediction')
        
        # Use original matrix without modification for imshow
        axs["Original_HiC"].imshow(self.pred_original, cmap=color_map_hic, vmin=0, vmax=3)
        axs["Original_HiC"].get_xaxis().set_ticks([])
        axs["Original_HiC"].get_yaxis().set_visible(False)
        
        # Add gray overlay rectangles for L-shaped region
        self.add_gray_overlay(axs["Original_HiC"])

        # Panel 2: Duplicated Hi-C prediction
        axs["Duplicated_HiC"].set_title(f'After tandem duplication (ChIP-seq mode: {self.chip_mode})')
        axs["Duplicated_HiC"].imshow(self.pred_duplicated, cmap=color_map_hic, vmin=0, vmax=3)
        axs["Duplicated_HiC"].get_xaxis().set_ticks([])
        axs["Duplicated_HiC"].get_yaxis().set_visible(False)

        # Panel 3: Difference Hi-C map
        color_map_diff = self.get_colormap_diff()
        axs["Difference_HiC"].set_title('Difference (Duplicated - Original)')
        axs["Difference_HiC"].imshow(self.diff_map, cmap=color_map_diff, vmin=-2, vmax=2)
        axs["Difference_HiC"].get_xaxis().set_ticks([])
        axs["Difference_HiC"].get_yaxis().set_visible(False)

        # Virtual 4C tracks (replacing ChIP-seq tracks)
        self.plot_virtual_4c_tracks(axs, source_start_genomic, source_end_genomic)

        # Add duplication region lines to Hi-C maps if requested
        if self.show_duplication_line:
            DASH_W = 3.0
            DASHES = (4, 4)
            
            # Convert genomic positions to pixel coordinates (similar to original code)
            # Assuming 256 pixels represents 2097152 bp
            scaling_factor = 256 / 2097152
            
            source_start_px = self.dup_source_start * scaling_factor
            source_end_px = (self.dup_source_start + self.dup_source_width) * scaling_factor
            
            # In tandem duplication mode, inserted region starts right after source
            insert_start_px = source_end_px
            insert_end_px = insert_start_px + (self.dup_source_width * scaling_factor)
            
            # Add lines to Hi-C panels only
            hic_panels = ['Original_HiC', 'Duplicated_HiC', 'Difference_HiC']
            for ax_name in hic_panels:
                ax = axs[ax_name]
                # Source region (blue lines) - show in all Hi-C panels
                ax.axvline(x=source_start_px, color='b', ls="--", dashes=DASHES, lw=DASH_W)
                ax.axvline(x=source_end_px, color='b', ls="--", dashes=DASHES, lw=DASH_W)
                ax.axhline(y=source_start_px, color='b', ls="--", dashes=DASHES, lw=DASH_W)
                ax.axhline(y=source_end_px, color='b', ls="--", dashes=DASHES, lw=DASH_W)
                
                # Inserted region (green lines) - show in duplicated and difference Hi-C panels
                if ax_name in ['Duplicated_HiC', 'Difference_HiC']:
                    ax.axvline(x=insert_start_px, color='g', ls="--", dashes=DASHES, lw=DASH_W)
                    ax.axvline(x=insert_end_px, color='g', ls="--", dashes=DASHES, lw=DASH_W)
                    ax.axhline(y=insert_start_px, color='g', ls="--", dashes=DASHES, lw=DASH_W)
                    ax.axhline(y=insert_end_px, color='g', ls="--", dashes=DASHES, lw=DASH_W)

        self.save_data(plt)

    def save_data(self, plt):
        """Save the plot and numerical data"""
        filename_base = f'{self.chr_name}_{self.start_pos}_tandem_dup_{self.dup_source_start}_{self.dup_source_width}_chip_{self.chip_mode}'
        
        # Save plot
        plt.savefig(f'{self.save_path}/imgs/{filename_base}.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # Save numerical data
        np.save(f'{self.save_path}/npy/{filename_base}_original', self.pred_original)
        np.save(f'{self.save_path}/npy/{filename_base}_duplicated', self.pred_duplicated)
        np.save(f'{self.save_path}/npy/{filename_base}_difference', self.diff_map)
        
        print(f"Results saved to {self.save_path}")
        print(f"  - Images: {filename_base}.png")
        print(f"  - Arrays: {filename_base}_original/duplicated/difference.npy")

if __name__ == '__main__':
    main() 
