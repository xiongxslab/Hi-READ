import os
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import plot_utils

import argparse


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
    parser = argparse.ArgumentParser(description='Hi-READ Editing Module.')
    
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

    # ============ 1. 原始预测（无删除）============
    window_original = 2097152
    seq_region_orig, chip_region_orig = infer.load_region(chr_name, 
            start, seq_path, chip_path, window = window_original)
    pred_original = infer.prediction(seq_region_orig, chip_region_orig, model_path)
    insulation_original = chr_score(pred_original, res=10000, radius=500000, pseudocount_coeff=30)
    
    # ============ 2. 删除后预测 ============
    window = 2097152 + deletion_width
    seq_region, chip_region = infer.load_region(chr_name, 
            start, seq_path, chip_path, window = window)
    # Delete inputs
    seq_region, chip_region = deletion_with_padding(start, 
            deletion_start, deletion_width, seq_region, chip_region, 
             end_padding_type)
    # Prediction
    pred_deletion = infer.prediction(seq_region, chip_region, model_path)
    insulation_deletion = chr_score(pred_deletion, res=10000, radius=500000, pseudocount_coeff=30)
    
    # ============ 3. 计算差异 ============
    pred_diff = pred_deletion - pred_original
    insulation_diff = insulation_deletion - insulation_original
    
    # ============ 4. 计算删除区域在 heatmap 中的像素位置 ============
    # 每个 bin 对应 8192 bp (2097152 / 256 = 8192)
    bin_size = 8192
    deletion_bin_start = int((deletion_start - start) / bin_size)
    deletion_bin_end = int((deletion_start - start + deletion_width) / bin_size)
    
    # Initialize plotting class with three-panel layout
    plot = plot_utils.MatrixPlotDeletionComparison(
        output_path=output_path,
        pred_original=pred_original,
        pred_deletion=pred_deletion,
        pred_diff=pred_diff,
        insulation_original=insulation_original,
        insulation_deletion=insulation_deletion,
        insulation_diff=insulation_diff,
        prefix='deletion_comparison',
        celltype=celltype,
        chr_name=chr_name,
        start_pos=start,
        deletion_start=deletion_start,
        deletion_width=deletion_width,
        deletion_bin_start=deletion_bin_start,
        deletion_bin_end=deletion_bin_end,
        padding_type=end_padding_type,
        show_deletion_line=show_deletion_line
    )
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

if __name__ == '__main__':
    main()
