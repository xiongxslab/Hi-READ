#!/usr/bin/env python3
"""
Universal FitHiChIP Loops caller script - parameterized version
Automatically converts npy files to bed files and runs FitHiChIP for loop calling
Supports complete workflow for any dataset
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
import numpy as np
from datetime import datetime
import json
import tempfile
import shutil


def env_or_none(env_name):
    value = os.environ.get(env_name, "").strip()
    return value or None

class UniversalFitHiChIPCaller:
    """
    Universal FitHiChIP loops caller - parameterized version
    """
    
    def __init__(self, npy_base_dir, output_dir, dataset_name, 
                 bin_size=8192, scaling_factor=10, peaks_file=None,
                 interaction_type=1, low_dist_thr=50000, upp_dist_thr=2000000,
                 fithichip_path=None,
                 macs2_output_dir=None,
                 npy_workers=None, keep_temp_files=False):
        
        self.npy_base_dir = npy_base_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.bin_size = bin_size
        self.scaling_factor = scaling_factor
        self.interaction_type = interaction_type
        self.low_dist_thr = low_dist_thr
        self.upp_dist_thr = upp_dist_thr
        self.fithichip_path = fithichip_path or env_or_none("HIREAD_FITHICHIP_DIR")
        self.macs2_output_dir = macs2_output_dir or env_or_none("HIREAD_MACS2_OUTPUT_DIR")
        self.npy_workers = npy_workers
        self.keep_temp_files = keep_temp_files
        self.temp_bed_dir = None
        
        # 自动构造peaks文件路径
        if peaks_file is None:
            self.peaks_file = self._find_peaks_file()
        else:
            self.peaks_file = peaks_file
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'dataset_name': dataset_name,
            'total_npy_files': 0,
            'converted_files': 0,
            'failed_conversions': 0,
            'bed_file_path': '',
            'fithichip_success': False,
            'processing_time': 0,
            'parameters': {
                'bin_size': bin_size,
                'scaling_factor': scaling_factor,
                'interaction_type': interaction_type,
                'low_dist_thr': low_dist_thr,
                'upp_dist_thr': upp_dist_thr,
                'npy_workers': npy_workers,
                'keep_temp_files': keep_temp_files,
                'peaks_file': self.peaks_file,
                'transformation': 'exp(x)-1'
            },
            'total_bed_entries': 0,
            'chunk_files': 0
        }
    
    def _find_peaks_file(self):
        """自动查找数据集对应的peaks文件"""
        peak_dir_aliases = {
            'h9_CTCF': 'CTCF',
            'h9_CTCF_NT': 'CTCF'
        }
        peak_dir_name = peak_dir_aliases.get(self.dataset_name, self.dataset_name)
        if not self.macs2_output_dir:
            print("未设置MACS2输出目录，将跳过自动peak文件查找")
            return None

        dataset_peak_dir = os.path.join(self.macs2_output_dir, peak_dir_name)
        
        # 定义已知的数据集与peak文件名的映射
        known_peak_files = {
            'CTCF': 'SRR6177938_peaks.narrowPeak',
            'h9_CTCF': 'SRR6177938_peaks.narrowPeak',
            'h9_CTCF_NT': 'SRR6177938_peaks.narrowPeak',
            'KLF4': 'SRR6177946_peaks.narrowPeak',
            'NANOG': 'SRR6177944_peaks.narrowPeak',
            'OCT4': 'SRR6177948_peaks.narrowPeak',
            'Rad21': 'SRR6177934_peaks.narrowPeak',
            'NTKO': 'GSM3791748_4-NTKO2-1_CTCF_noinput_peaks.bed.gz',
            'TKO': 'GSM3791750_6-TKO1-1_CTCF_noinput_peaks.bed.gz'
        }
        
        # 首先尝试已知的映射
        peak_filename = known_peak_files.get(self.dataset_name) or known_peak_files.get(peak_dir_name)
        if peak_filename:
            peak_file = os.path.join(dataset_peak_dir, peak_filename)
            if os.path.exists(peak_file):
                if peak_dir_name != self.dataset_name:
                    print(f"使用peak目录别名: {self.dataset_name} -> {peak_dir_name}")
                print(f"找到已知的peak文件: {peak_file}")
                return peak_file
        
        # 如果没有已知映射或文件不存在，搜索目录中的常见peak文件
        if os.path.exists(dataset_peak_dir):
            peak_candidates = []
            for file in sorted(os.listdir(dataset_peak_dir)):
                if file.endswith(('.narrowPeak', '.bed.gz', '.bed')):
                    peak_candidates.append(file)

            if peak_candidates:
                peak_file = os.path.join(dataset_peak_dir, peak_candidates[0])
                print(f"自动找到peak文件: {peak_file}")
                if len(peak_candidates) > 1:
                    print(f"注意: 找到多个peak文件，使用: {peak_candidates[0]}")
                    print(f"其他文件: {peak_candidates[1:]}")
                return peak_file
        
        # 如果都找不到，返回None
        print(f"警告: 未找到数据集 {self.dataset_name} 的peak文件")
        print(f"搜索目录: {dataset_peak_dir}")
        return None
    
    def find_npy_files(self):
        """查找所有npy文件 - 支持多种格式，优先选择_final.npy"""
        print(f"正在查找{self.dataset_name} npy文件...")
        print(f"搜索目录: {self.npy_base_dir}")
        
        # 使用字典来跟踪每个染色体位置的文件，优先保留_final.npy
        npy_files_dict = {}
        total_found_files = 0
        replaced_files = 0
        replacement_examples = []
        
        if not os.path.exists(self.npy_base_dir):
            raise FileNotFoundError(f"NPY基础目录不存在: {self.npy_base_dir}")
        
        # 遍历所有子目录查找npy文件
        for root, dirs, files in os.walk(self.npy_base_dir):
            for file in files:
                # 支持多种npy文件格式
                if self._is_valid_npy_file(file):
                    full_path = os.path.join(root, file)
                    total_found_files += 1
                    
                    # 解析文件名获取染色体和位置
                    chr_name, region_start = self._parse_npy_filename(file)
                    if chr_name and region_start is not None:
                        key = f"{chr_name}_{region_start}"
                        
                        # 如果这个位置还没有文件，或者当前文件是_final.npy格式
                        if key not in npy_files_dict:
                            npy_files_dict[key] = full_path
                        elif file.endswith('_final.npy') and not npy_files_dict[key].endswith('_final.npy'):
                            # 如果当前文件是_final.npy而之前的不是，则替换
                            old_file = os.path.basename(npy_files_dict[key])
                            npy_files_dict[key] = full_path
                            replaced_files += 1
                            if len(replacement_examples) < 20:
                                replacement_examples.append((file, old_file))
        
        # 转换为列表
        npy_files = list(npy_files_dict.values())
                    
        print(f"扫描到 {total_found_files} 个npy文件，去重后保留 {len(npy_files)} 个")
        if replaced_files > 0:
            print(f"替换了 {replaced_files} 个文件 (_final.npy 优先)")
            for new_file, old_file in replacement_examples:
                print(f"  优先选择示例: {new_file} (替换 {old_file})")
            if replaced_files > len(replacement_examples):
                print(f"  ... 其余 {replaced_files - len(replacement_examples)} 个替换已省略")
        
        if len(npy_files) > 0:
            print("最终文件格式分布:")
            final_count = sum(1 for f in npy_files if f.endswith('_final.npy'))
            regular_count = len(npy_files) - final_count
            print(f"  • _final.npy 格式: {final_count} 个")
            print(f"  • .npy 格式: {regular_count} 个")
            print("  • 优先级规则: _final.npy > .npy (同位置时)")
            
        self.stats['total_npy_files'] = len(npy_files)
        
        return sorted(npy_files)
    
    def _is_valid_npy_file(self, filename):
        """检查是否是有效的npy文件格式"""
        if not filename.endswith('.npy'):
            return False
            
        # 移除.npy后缀进行格式检查
        basename = filename[:-4]  # 移除.npy
        
        # 处理_final.npy格式
        if basename.endswith('_final'):
            basename = basename[:-6]  # 移除_final
        
        # 检查是否符合 chrX_Y 格式
        parts = basename.split('_')
        if len(parts) >= 2:
            chr_part = parts[0]
            pos_part = parts[1]
            
            # 检查染色体格式 (chr开头) 和位置是否为数字
            if chr_part.startswith('chr') and pos_part.isdigit():
                return True
                
        return False
     
    def _parse_npy_filename(self, filename):
        """解析npy文件名，支持多种格式
        
        支持的格式:
        - chr1_0_final.npy (RRTdiffusion格式)
        - chr1_0.npy (Real数据格式)
        
        返回: (chr_name, region_start) 或 (None, None)
        """
        if not filename.endswith('.npy'):
            return None, None
            
        # 移除.npy后缀
        basename = filename[:-4]
        
        # 处理_final.npy格式
        if basename.endswith('_final'):
            basename = basename[:-6]  # 移除_final
        
        # 解析chrX_Y格式
        parts = basename.split('_')
        if len(parts) >= 2:
            chr_name = parts[0]
            try:
                region_start = int(parts[1])
                # 验证染色体格式
                if chr_name.startswith('chr'):
                    return chr_name, region_start
            except ValueError:
                pass
                
        return None, None
     
    def convert_single_npy_to_bed(self, npy_file_path):
        """转换单个npy文件到临时bed分块文件"""
        try:
            # 解析文件名获取染色体和起始位置
            filename = os.path.basename(npy_file_path)
            chr_name, region_start = self._parse_npy_filename(filename)
            
            if chr_name is None or region_start is None:
                return {
                    'chunk_file': None,
                    'entry_count': 0,
                    'source_file': npy_file_path,
                    'error': f"无法解析文件名: {filename}"
                }
            
            # 加载npy数据
            data = np.load(npy_file_path)
            matrix_size = data.shape[0]
            chunk_basename = f"{chr_name}_{region_start}.bed"
            chunk_file = os.path.join(self.temp_bed_dir, chunk_basename)
            entry_count = 0
            
            with open(chunk_file, 'w') as chunk_f:
                for i in range(matrix_size):
                    start1 = region_start + i * self.bin_size
                    end1 = start1 + self.bin_size
                    
                    for j in range(i, matrix_size):  # 只处理上三角矩阵
                        contact_count = data[i, j]
                        if contact_count <= 0:
                            continue
                        
                        # 计算基因组坐标
                        start2 = region_start + j * self.bin_size
                        end2 = start2 + self.bin_size
                        
                        # 应用exp(x)-1变换，然后缩放并转换为整数
                        transformed_count = np.exp(contact_count) - 1
                        scaled_count = int(transformed_count * self.scaling_factor + 0.5)
                        
                        if scaled_count > 0:
                            chunk_f.write(
                                f"{chr_name}\t{start1}\t{end1}\t"
                                f"{chr_name}\t{start2}\t{end2}\t{scaled_count}\n"
                            )
                            entry_count += 1
            
            if entry_count == 0:
                os.remove(chunk_file)
                chunk_file = None
            
            return {
                'chunk_file': chunk_file,
                'entry_count': entry_count,
                'source_file': npy_file_path,
                'error': None
            }
            
        except Exception as e:
            return {
                'chunk_file': None,
                'entry_count': 0,
                'source_file': npy_file_path,
                'error': str(e)
            }
    
    def convert_npy_to_bed_parallel(self, npy_files):
        """并行转换npy文件到bed格式 - 支持多种格式"""
        print(f"开始并行转换 {len(npy_files)} 个npy文件到bed格式...")
        print(f"缩放因子: {self.scaling_factor}, bin大小: {self.bin_size}")
        print(f"支持的格式: chr*_*_final.npy 和 chr*_*.npy")
        
        # 使用多进程并行处理
        if self.npy_workers is not None:
            max_workers = max(1, int(self.npy_workers))
        else:
            max_workers = min(multiprocessing.cpu_count(), 8)
        print(f"使用 {max_workers} 个并行进程")
        self.temp_bed_dir = tempfile.mkdtemp(
            prefix=f"{self.dataset_name}_bed_chunks_",
            dir=self.output_dir
        )
        print(f"临时bed分块目录: {self.temp_bed_dir}")
        
        chunk_files = []
        failed_files = []
        total_bed_entries = 0
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.map(self.convert_single_npy_to_bed, npy_files)
        
        # 收集结果
        for i, result in enumerate(results):
            if result['error'] is None:
                if result['chunk_file']:
                    chunk_files.append(result['chunk_file'])
                total_bed_entries += result['entry_count']
                self.stats['converted_files'] += 1
            else:
                failed_files.append({
                    'file': result['source_file'],
                    'error': result['error']
                })
                self.stats['failed_conversions'] += 1
            
            # 显示进度
            if (i + 1) % 100 == 0 or i + 1 == len(npy_files):
                print(f"进度: {i+1}/{len(npy_files)} ({(i+1)/len(npy_files)*100:.1f}%)")
        
        print(f"转换完成: {self.stats['converted_files']} 成功, {self.stats['failed_conversions']} 失败")
        print(f"生成bed条目总数: {total_bed_entries:,}")
        print(f"生成bed分块文件: {len(chunk_files)} 个")
        self.stats['total_bed_entries'] = total_bed_entries
        self.stats['chunk_files'] = len(chunk_files)
        
        if failed_files:
            print(f"失败的文件 (前5个):")
            for failed in failed_files[:5]:
                print(f"  • {os.path.basename(failed['file'])}: {failed['error']}")
        
        return sorted(chunk_files)
    
    def save_bed_file(self, bed_chunk_files):
        """保存bed文件并排序"""
        if not bed_chunk_files:
            raise ValueError("没有bed分块文件可保存")
        
        print(f"正在合并和排序 {len(bed_chunk_files)} 个bed分块文件...")

        final_bed_file = os.path.join(self.output_dir, f"{self.dataset_name}_genome_wide.bed")
        
        # 直接将分块流式写入sort，避免额外生成一个巨大的unsorted.bed。
        print("使用系统sort命令进行流式最终排序...")
        sort_parallel = min(multiprocessing.cpu_count(), 4)
        sort_cmd = [
            "sort",
            f"--parallel={sort_parallel}",
            "-T",
            self.output_dir,
            "-k1,1",
            "-k2,2n",
            "-k4,4",
            "-k5,5n",
        ]

        try:
            with open(final_bed_file, 'wb') as sorted_f:
                sort_proc = subprocess.Popen(sort_cmd, stdin=subprocess.PIPE, stdout=sorted_f)
                try:
                    for i, chunk_file in enumerate(bed_chunk_files, 1):
                        with open(chunk_file, 'rb') as chunk_f:
                            shutil.copyfileobj(chunk_f, sort_proc.stdin, length=1024 * 1024)

                        if not self.keep_temp_files:
                            os.remove(chunk_file)

                        if i % 200 == 0 or i == len(bed_chunk_files):
                            print(f"分块流式进度: {i}/{len(bed_chunk_files)}")
                finally:
                    if sort_proc.stdin:
                        sort_proc.stdin.close()

                return_code = sort_proc.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, sort_cmd)
        except Exception:
            if os.path.exists(final_bed_file):
                os.remove(final_bed_file)
            raise

        print(f"bed文件已保存: {final_bed_file}")
        
        if not self.keep_temp_files:
            if self.temp_bed_dir and os.path.exists(self.temp_bed_dir):
                shutil.rmtree(self.temp_bed_dir)
        else:
            if self.temp_bed_dir:
                print(f"保留临时bed分块目录: {self.temp_bed_dir}")
        
        self.stats['bed_file_path'] = final_bed_file
        return final_bed_file
    
    def create_fithichip_config(self, bed_file):
        """创建FitHiChIP配置文件"""
        config_file = os.path.join(self.output_dir, f"{self.dataset_name}_fithichip_config.conf")
        
        # 确定交互类型和peaks文件设置
        if self.interaction_type in [1, 2, 3] and self.peaks_file and os.path.exists(self.peaks_file):
            peaks_line = f"PeakFile={self.peaks_file}"
            peaks_comment = ""
            interaction_names = {1: "peak-to-peak", 2: "peak-to-non-peak", 3: "peak-to-all"}
            print(f"使用{interaction_names[self.interaction_type]}模式，peak文件: {self.peaks_file}")
        else:
            peaks_line = "PeakFile="
            peaks_comment = "#"
            if self.interaction_type in [1, 2, 3]:
                print(f"警告: peak文件不存在或无效，自动切换到all-to-all模式")
                self.interaction_type = 4  # 强制设为all-to-all
        
        chrom_sizes_file = env_or_none("HIREAD_HG38_CHROM_SIZES")
        if not chrom_sizes_file:
            if not self.fithichip_path:
                raise ValueError(
                    "FitHiChIP路径未设置，且未提供HIREAD_HG38_CHROM_SIZES，无法生成配置文件"
                )
            chrom_sizes_file = os.path.join(self.fithichip_path, "TestData", "hg38.chrom.sizes")

        config_content = f"""##********
## {self.dataset_name}数据FitHiChIP配置文件 - 通用参数化版本
##********

##********
## 输入文件配置
##********
Bed={bed_file}

##********
## 参考文件
##********
ChrSizeFile={chrom_sizes_file}

##********
## 输出目录
##********
OutDir={self.output_dir}/{self.dataset_name}_fithichip_results

##********
## 基本参数
##********
CircularGenome=0

##********
## 使用peaks文件进行loop calling (如果提供)
##********
{peaks_comment}{peaks_line}

##********
## 互作类型
##********
IntType={self.interaction_type}

## Bin size (bp)
BINSIZE={self.bin_size}

##********
## 距离阈值设置
##********
LowDistThr={self.low_dist_thr}
UppDistThr={self.upp_dist_thr}

##********
## 背景模型参数
##********
UseP2PBackgrnd=0

##********
## Bias correction
##********
BiasType=1

##********
## 合并相邻互作
##********
MergeInt=1

##********
## Q-value threshold for significant interactions
##********
QVALUE=0.01

##********
## Prefix for output files
##********
PREFIX=FitHiChIP_{self.dataset_name}

##********
## 覆盖现有文件
##********
OverWrite=1
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"FitHiChIP配置文件已创建: {config_file}")
        return config_file
    
    def run_fithichip(self, config_file):
        """运行FitHiChIP"""
        print("=" * 60)
        print(f"开始运行FitHiChIP for {self.dataset_name}")
        print("=" * 60)

        if not self.fithichip_path:
            raise ValueError("未设置FitHiChIP路径，请传入 --fithichip-path 或设置 HIREAD_FITHICHIP_DIR")

        # 检查FitHiChIP路径
        fithichip_script = os.path.join(self.fithichip_path, "FitHiChIP_HiCPro.sh")
        if not os.path.exists(fithichip_script):
            raise FileNotFoundError(f"FitHiChIP脚本不存在: {fithichip_script}")

        activate_cmd = env_or_none("HIREAD_FITHICHIP_ACTIVATE")
        if activate_cmd:
            shell_cmd = f"{activate_cmd} && cd {self.fithichip_path} && bash FitHiChIP_HiCPro.sh -C {config_file}"
        else:
            shell_cmd = f"cd {self.fithichip_path} && bash FitHiChIP_HiCPro.sh -C {config_file}"
        cmd = ["bash", "-lc", shell_cmd]
        
        print(f"FitHiChIP命令: {' '.join(cmd)}")
        print(f"配置文件: {config_file}")
        
        # 运行FitHiChIP
        try:
            start_time = datetime.now()
            
            # 创建日志文件
            log_file = os.path.join(self.output_dir, f"{self.dataset_name}_fithichip.log")
            
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                print(f"FitHiChIP进程已启动 (PID: {process.pid})")
                print(f"日志文件: {log_file}")
                print("实时输出:")
                print("-" * 40)
                
                # 实时显示输出
                for line in process.stdout:
                    print(line.rstrip())
                    log_f.write(line)
                    log_f.flush()
                
                process.wait()
                return_code = process.returncode
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("-" * 40)
            print(f"FitHiChIP完成，返回码: {return_code}")
            print(f"运行时间: {duration/60:.1f} 分钟")
            
            if return_code == 0:
                self.stats['fithichip_success'] = True
                print("✅ FitHiChIP运行成功!")
            else:
                print("❌ FitHiChIP运行失败!")
                
            return return_code == 0
            
        except Exception as e:
            print(f"运行FitHiChIP时出错: {str(e)}")
            return False
    
    def run_complete_pipeline(self):
        """运行完整的pipeline"""
        start_time = datetime.now()
        print("=" * 80)
        print(f"=== {self.dataset_name} FitHiChIP完整流程开始 ===")
        print("=" * 80)
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集: {self.dataset_name}")
        print(f"NPY目录: {self.npy_base_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"参数: bin_size={self.bin_size}, scaling_factor={self.scaling_factor}")
        print(f"交互类型: {self.interaction_type}")
        if self.peaks_file:
            print(f"Peaks文件: {self.peaks_file}")
        
        try:
            # 步骤1: 查找npy文件
            print("\n" + "="*50)
            print("步骤1: 查找npy文件")
            print("="*50)
            npy_files = self.find_npy_files()
            
            if not npy_files:
                raise ValueError("未找到任何npy文件")
            
            # 步骤2: 转换npy到bed
            print("\n" + "="*50)
            print("步骤2: 转换npy文件到bed格式")
            print("="*50)
            bed_chunk_files = self.convert_npy_to_bed_parallel(npy_files)
            
            if not bed_chunk_files:
                raise ValueError("未能生成任何bed分块文件")
            
            # 步骤3: 保存bed文件
            print("\n" + "="*50)
            print("步骤3: 保存和排序bed文件")
            print("="*50)
            bed_file = self.save_bed_file(bed_chunk_files)
            
            # 步骤4: 创建FitHiChIP配置
            print("\n" + "="*50)
            print("步骤4: 创建FitHiChIP配置文件")
            print("="*50)
            config_file = self.create_fithichip_config(bed_file)
            
            # 步骤5: 运行FitHiChIP
            print("\n" + "="*50)
            print("步骤5: 运行FitHiChIP")
            print("="*50)
            success = self.run_fithichip(config_file)
            
            # 总结
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            self.stats['processing_time'] = total_duration
            
            print("\n" + "="*80)
            print(f"=== {self.dataset_name} FitHiChIP流程完成 ===")
            print("="*80)
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总处理时间: {total_duration/60:.1f} 分钟")
            print(f"处理的npy文件: {self.stats['converted_files']}/{self.stats['total_npy_files']}")
            print(f"生成的bed文件: {bed_file}")
            print(f"FitHiChIP状态: {'成功' if success else '失败'}")
            
            # 保存统计信息
            stats_file = os.path.join(self.output_dir, f"{self.dataset_name}_pipeline_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
            print(f"统计信息已保存: {stats_file}")
            
            return success
            
        except Exception as e:
            if self.temp_bed_dir and os.path.exists(self.temp_bed_dir) and not self.keep_temp_files:
                try:
                    shutil.rmtree(self.temp_bed_dir)
                except OSError:
                    pass
            print(f"\n❌ 流程执行失败: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='通用FitHiChIP Loops调用工具')
    
    # 必需参数
    parser.add_argument('--npy-dir', required=True,
                       help='包含.npy矩阵文件的基础目录')
    parser.add_argument('--output-dir', required=True,
                       help='输出目录')
    parser.add_argument('--dataset-name', required=True,
                       help='数据集名称')
    
    # 可选参数
    parser.add_argument('--bin-size', type=int, default=8192,
                       help='Bin大小 (默认: 8192)')
    parser.add_argument('--scaling-factor', type=float, default=10,
                       help='接触计数缩放因子 (默认: 10)')
    parser.add_argument('--peaks-file', 
                       help='Peaks文件路径 (用于peak-based交互类型)')
    parser.add_argument('--interaction-type', type=int, default=1,
                       help='交互类型: 1=peak to peak, 2=peak to non peak, 3=peak to all, 4=all to all (默认: 1)')
    parser.add_argument('--low-dist-thr', type=int, default=20000,
                       help='最小距离阈值 (默认: 8192)')
    parser.add_argument('--upp-dist-thr', type=int, default=2000000,
                       help='最大距离阈值 (默认: 2000000)')
    parser.add_argument('--fithichip-path', default=env_or_none('HIREAD_FITHICHIP_DIR'),
                       help='FitHiChIP安装路径，或通过 HIREAD_FITHICHIP_DIR 提供')
    parser.add_argument('--macs2-output-dir', default=env_or_none('HIREAD_MACS2_OUTPUT_DIR'),
                       help='MACS2输出目录，用于自动查找peak文件，也可通过 HIREAD_MACS2_OUTPUT_DIR 提供')
    parser.add_argument('--npy-workers', type=int,
                       help='npy转bed时使用的并行进程数 (默认: min(CPU, 8))')
    parser.add_argument('--keep-temp-files', action='store_true',
                       help='保留中间bed分块文件和未排序bed文件')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.npy_dir):
        print(f"错误: NPY目录不存在: {args.npy_dir}")
        sys.exit(1)
    
    if not args.fithichip_path:
        print("错误: 未设置FitHiChIP路径，请传入 --fithichip-path 或设置 HIREAD_FITHICHIP_DIR")
        sys.exit(1)

    if not os.path.exists(args.fithichip_path):
        print(f"错误: FitHiChIP路径不存在: {args.fithichip_path}")
        sys.exit(1)
    
    if args.peaks_file and not os.path.exists(args.peaks_file):
        print(f"错误: Peaks文件不存在: {args.peaks_file}")
        sys.exit(1)
    
    # 验证macs2输出目录
    if args.macs2_output_dir and not os.path.exists(args.macs2_output_dir):
        print(f"警告: MACS2输出目录不存在: {args.macs2_output_dir}")
        print("将尝试继续运行，但可能无法找到peak文件")
    
    print("通用FitHiChIP Loops调用工具")
    print(f"数据集: {args.dataset_name}")
    print(f"NPY目录: {args.npy_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"参数: bin_size={args.bin_size}, scaling_factor={args.scaling_factor}")
    print(f"交互类型: {args.interaction_type}")
    if args.peaks_file:
        print(f"Peaks文件: {args.peaks_file}")
    
    # 创建caller并运行
    caller = UniversalFitHiChIPCaller(
        npy_base_dir=args.npy_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        bin_size=args.bin_size,
        scaling_factor=args.scaling_factor,
        peaks_file=args.peaks_file,
        interaction_type=args.interaction_type,
        low_dist_thr=args.low_dist_thr,
        upp_dist_thr=args.upp_dist_thr,
        fithichip_path=args.fithichip_path,
        macs2_output_dir=args.macs2_output_dir,
        npy_workers=args.npy_workers,
        keep_temp_files=args.keep_temp_files
    )
    
    # 运行完整流程
    success = caller.run_complete_pipeline()
    
    if success:
        print("\n🎉 FitHiChIP流程成功完成!")
        sys.exit(0)
    else:
        print("\n❌ FitHiChIP流程失败!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
