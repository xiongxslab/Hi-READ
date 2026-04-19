import hiread.inference.utils.inference_utils as infer
from hiread.inference.utils import plot_utils
import argparse
import sys
import numpy as np

def shuffle_sequence(seq, seed=None):
    """
    随机打乱 seq 数据，包括行和列
    Args:
        seq: 序列数据 (numpy array)，形状为 (num_samples, num_features)。
        seed: 随机种子，默认为 None。设置固定值以确保结果可重复。
    Returns:
        shuffled_seq: 行和列完全打乱后的序列
    """
    if seed is not None:
        np.random.seed(seed)  # 设置随机种子以保证可重复性

    flattened = seq.flatten()  # 将数组展平为一维
    np.random.shuffle(flattened)  # 对一维数组进行随机打乱
    shuffled_seq = flattened.reshape(seq.shape)  # 恢复为原始形状

    return shuffled_seq


def main():
    parser = argparse.ArgumentParser(description='Hi-READ Prediction Module.')
    
    # 添加新的参数 --shuffle-sequence，默认值为 False
    parser.add_argument('--shuffle', dest='shuffle_sequence', action='store_true',
                        help='Shuffle the input sequence before prediction (default: False)')
    parser.add_argument('--seed', dest='shuffle_seed', type=int, default=None,
                        help='Random seed for shuffling the sequence (default: None)')
    
    parser.add_argument('--out', dest='output_path', default='outputs',
                        help='Output path for storing results (default: %(default)s)')
    parser.add_argument('--celltype', dest='celltype', 
                        help='Sample cell type for prediction, used for output separation')
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

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    # 调用 single_prediction，并传入 shuffle_sequence 和 shuffle_seed 参数
    single_prediction(args.output_path, args.celltype, 
                      args.chr_name, args.start,
                      args.model_path, 
                      args.seq_path, args.chip_path,
                      args.shuffle_sequence, args.shuffle_seed)

def single_prediction(output_path, celltype, chr_name, start, model_path, seq_path, chip_path, shuffle_sequence_flag=False, shuffle_seed=None):
    # 加载区域数据
    seq_region, chip_region = infer.load_region(chr_name, start, seq_path, chip_path)
    print("seq_region shape:", seq_region.shape)
    # 如果 shuffle_sequence_flag 为 True，则打乱序列
    if shuffle_sequence_flag:
        print("Shuffling the sequence...")
        seq_region = shuffle_sequence(seq_region, seed=shuffle_seed)  # 使用传入的随机种子进行打乱

    # 模型预测
    pred = infer.prediction(seq_region, chip_region, model_path)

    # 绘图和保存结果
    plot = plot_utils.MatrixPlot(output_path, pred, 'prediction', celltype, chr_name, start)
    plot.plot()

    pred = np.array(pred)
    np.save(output_path + '/prediction.npy', pred)

if __name__ == '__main__':
    main()
