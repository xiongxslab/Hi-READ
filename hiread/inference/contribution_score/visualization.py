#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional

class ComprehensiveVisualizer:
    """综合可视化器"""
    
    def __init__(self, chr_name: str, start_pos: int, save_path: str):
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.save_path = save_path
        self._configure_vector_text_export()

    def _configure_vector_text_export(self):
        """确保PDF/SVG导出时文字保持为可编辑矢量文本，不转曲。"""
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['text.usetex'] = False

    def _normalize_track(self, track_data: np.ndarray) -> np.ndarray:
        """将1D track按min-max归一化到[0, 1]，避免不同层量纲不可比。"""
        data = np.asarray(track_data, dtype=np.float32)
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        denom = data_max - data_min
        if denom < 1e-12:
            return np.zeros_like(data, dtype=np.float32)
        return (data - data_min) / (denom + 1e-12)
        
    def get_colormap_hic(self):
        """获取Hi-C热图的colormap - 从白色到鲜红色"""
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1.0,1.0,1.0),(1.0,0.0,0.0)])
        return color_map
        
    def create_paper_style_visualization(self, 
                                       prediction: np.ndarray,
                                       chipseq_data: np.ndarray,
                                       gradcam_1d: Optional[np.ndarray] = None,
                                       attention_weights: Optional[Dict] = None,
                                       impact_scores: Optional[np.ndarray] = None,
                                       perturb_starts: Optional[np.ndarray] = None,
                                       step_size: int = 8192):
        """创建类似论文图片的综合可视化"""
        
        # 计算track数量
        n_tracks = 2  # Hi-C + ChIP-seq
        if gradcam_1d is not None:
            n_tracks += 1
        if attention_weights is not None:
            n_tracks += len(attention_weights)
        if impact_scores is not None:
            n_tracks += 1
        
        # 创建图形，使用gridspec来精确控制布局
        fig = plt.figure(figsize=(12, 16))
        
        # 设置基因组坐标范围（Mb）
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        
        # 先创建Hi-C图，然后获取其精确位置用于对齐其他track
        matrix_size = prediction.shape[0]
        
        # 1. Hi-C预测热图 (顶部) - 创建临时GridSpec
        temp_gs = gridspec.GridSpec(1, 1, top=0.9, bottom=0.5)
        ax1 = fig.add_subplot(temp_gs[0])
        
        # 使用正方形比例显示Hi-C矩阵
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=2, 
                        aspect='equal', interpolation='bilinear', origin='upper')
        
        # 设置标题和标签
        ax1.set_title(f'{self.chr_name}: {genomic_start_mb:.1f}-{genomic_end_mb:.1f} Mb', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Genomic distance (Mb)', fontsize=12)
        
        # 设置Hi-C的刻度，对应基因组位置
        self._format_hic_ticks(ax1, matrix_size, genomic_start_mb, genomic_end_mb)
        
        # 绘制图形以获取Hi-C图的精确位置
        fig.canvas.draw()
        hic_pos = ax1.get_position()
        
        # 存储Hi-C图的位置信息
        hic_left = hic_pos.x0
        hic_right = hic_pos.x1  
        hic_width = hic_pos.width
        hic_bottom = hic_pos.y0
        
        track_idx = 1
        
        # 2. ChIP-seq track - 手动设置位置与Hi-C对齐
        track_height = 0.08
        track_spacing = 0.02
        current_y = hic_bottom - track_spacing - track_height
        
        ax2 = fig.add_axes([hic_left, current_y, hic_width, track_height])
        # 使用像素坐标与Hi-C对齐
        pixel_positions = np.linspace(0, matrix_size-1, len(chipseq_data))
        ax2.fill_between(pixel_positions, chipseq_data, alpha=0.7, color='blue', linewidth=0)
        ax2.plot(pixel_positions, chipseq_data, 'b-', linewidth=0.8)
        
        ax2.set_ylabel('ChIP-seq', fontsize=12)
        ax2.set_xlim(0, matrix_size-1)
        # 设置与Hi-C相同的x轴刻度
        self._format_track_ticks(ax2, matrix_size, genomic_start_mb, genomic_end_mb)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        current_y -= (track_height + track_spacing)
        track_idx += 1
        
        # 3. Grad-CAM track
        if gradcam_1d is not None:
            ax_grad = fig.add_axes([hic_left, current_y, hic_width, track_height])
            # 使用像素坐标与Hi-C对齐
            gradcam_pixel_positions = np.linspace(0, matrix_size-1, len(gradcam_1d))
            ax_grad.fill_between(gradcam_pixel_positions, gradcam_1d, alpha=0.7, color='cyan', linewidth=0)
            ax_grad.plot(gradcam_pixel_positions, gradcam_1d, 'c-', linewidth=0.8)
            
            ax_grad.set_ylabel('Grad-CAM', fontsize=12)
            ax_grad.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax_grad, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_grad.grid(True, alpha=0.3)
            ax_grad.spines['top'].set_visible(False)
            ax_grad.spines['right'].set_visible(False)
            
            current_y -= (track_height + track_spacing)
            track_idx += 1
        
        # 4. Attention权重tracks
        if attention_weights is not None:
            colors = ['orange', 'purple', 'brown', 'pink', 'green', 'red']
            color_idx = 0
            
            for attn_name, attn_data in attention_weights.items():
                if attn_data is not None:
                    # 新的attention数据结构是字典
                    if isinstance(attn_data, dict):
                        # 优先使用1D attention和regional attention进行可视化
                        attention_tracks = {}
                        
                        # 1. 添加1D attention track（优先使用256长度的数据）
                        if 'attention_1d_256' in attn_data:
                            attention_tracks[f'{attn_name}_1D'] = attn_data['attention_1d_256']
                        elif 'attention_1d' in attn_data:
                            attention_tracks[f'{attn_name}_1D'] = attn_data['attention_1d']
                        
                        # 2. 添加regional attention track (如果存在)
                        if 'regional_attention' in attn_data:
                            attention_tracks[f'{attn_name}_Regional'] = attn_data['regional_attention']
                        
                        # 如果都不存在，尝试从其他类型生成1D
                        if not attention_tracks:
                            if 'global_averaged' in attn_data:
                                # 从2D attention矩阵生成1D
                                attn_2d = attn_data['global_averaged']
                                attn_1d = np.mean(attn_2d, axis=0)
                                attention_tracks[f'{attn_name}_Global'] = attn_1d
                            elif 'head_averaged' in attn_data:
                                attn_2d = attn_data['head_averaged']
                                attn_1d = np.mean(attn_2d, axis=0)
                                attention_tracks[f'{attn_name}_HeadAvg'] = attn_1d
                        
                        # 绘制每个attention track
                        for track_name, track_data in attention_tracks.items():
                            if track_data is not None and len(track_data) > 0:
                                ax_attn = fig.add_axes([hic_left, current_y, hic_width, track_height])
                                
                                # 处理数据长度与矩阵大小的对齐
                                if len(track_data) == matrix_size:
                                    # 直接对齐
                                    attn_pixel_positions = np.arange(matrix_size)
                                elif len(track_data) == 16:
                                    # Regional attention - 扩展到matrix_size
                                    attn_pixel_positions = np.linspace(0, matrix_size-1, len(track_data))
                                else:
                                    # 其他情况 - 插值对齐
                                    attn_pixel_positions = np.linspace(0, matrix_size-1, len(track_data))
                                
                                color = colors[color_idx % len(colors)]
                                ax_attn.fill_between(attn_pixel_positions, track_data, alpha=0.7, color=color, linewidth=0)
                                ax_attn.plot(attn_pixel_positions, track_data, color=color, linewidth=0.8)
                                
                                ax_attn.set_ylabel(f'Attention\n{track_name}', fontsize=10)
                                ax_attn.set_xlim(0, matrix_size-1)
                                self._format_track_ticks(ax_attn, matrix_size, genomic_start_mb, genomic_end_mb)
                                ax_attn.grid(True, alpha=0.3)
                                ax_attn.spines['top'].set_visible(False)
                                ax_attn.spines['right'].set_visible(False)
                                
                                current_y -= (track_height + track_spacing)
                                track_idx += 1
                                color_idx += 1
                    
                    else:
                        # 兼容旧格式的数据
                        ax_attn = fig.add_axes([hic_left, current_y, hic_width, track_height])
                        
                        # 处理attention数据
                        if len(attn_data.shape) > 1:
                            attn_1d = attn_data.mean(axis=0)
                        else:
                            attn_1d = attn_data
                        
                        # 使用像素坐标与Hi-C对齐
                        attn_pixel_positions = np.linspace(0, matrix_size-1, len(attn_1d))
                        
                        color = colors[color_idx % len(colors)]
                        ax_attn.fill_between(attn_pixel_positions, attn_1d, alpha=0.7, color=color, linewidth=0)
                        ax_attn.plot(attn_pixel_positions, attn_1d, color=color, linewidth=0.8)
                        
                        ax_attn.set_ylabel(f'Attention\n{attn_name}', fontsize=12)
                        ax_attn.set_xlim(0, matrix_size-1)
                        self._format_track_ticks(ax_attn, matrix_size, genomic_start_mb, genomic_end_mb)
                        ax_attn.grid(True, alpha=0.3)
                        ax_attn.spines['top'].set_visible(False)
                        ax_attn.spines['right'].set_visible(False)
                        
                        current_y -= (track_height + track_spacing)
                        track_idx += 1
                        color_idx += 1
        
        # 5. Impact scores track (底部)
        if impact_scores is not None and perturb_starts is not None:
            ax_impact = fig.add_axes([hic_left, current_y, hic_width, track_height])
            
            # 将基因组坐标转换为像素坐标与Hi-C对齐
            # 计算每个perturbation在像素坐标系中的位置
            genomic_range = 2097152  # 2MB in bp
            impact_pixel_positions = []
            for start_pos in perturb_starts:
                # 计算相对于区域起始的偏移
                relative_pos = start_pos - self.start_pos
                # 转换为像素坐标
                pixel_pos = (relative_pos / genomic_range) * matrix_size
                impact_pixel_positions.append(pixel_pos)
            
            impact_pixel_positions = np.array(impact_pixel_positions)
            
            # 计算bar的宽度（以像素为单位）
            pixel_width = (step_size / genomic_range) * matrix_size * 0.8
            
            # 创建bar plot
            ax_impact.bar(impact_pixel_positions, impact_scores, 
                         width=pixel_width, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax_impact.set_ylabel('Impact', fontsize=12)
            ax_impact.set_xlabel('Genomic Position (Mb)', fontsize=12)
            ax_impact.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax_impact, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_impact.grid(True, alpha=0.3)
            ax_impact.spines['top'].set_visible(False)
            ax_impact.spines['right'].set_visible(False)
        
        # 所有track创建完成后，添加colorbar
        # 创建colorbar，高度与Hi-C图完全一致
        cbar_ax = fig.add_axes([hic_right + 0.02, hic_pos.y0, 0.02, hic_pos.height])
        cbar1 = fig.colorbar(im1, cax=cbar_ax)
        cbar1.set_label('Contact Frequency', fontsize=10)
        
        # 保存图形
        save_name = f'{self.chr_name}_{self.start_pos}_paper_style'
        plt.savefig(f'{self.save_path}/imgs/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.save_path}/imgs/{save_name}.svg', bbox_inches='tight')
        
        print(f"论文样式可视化保存到: {self.save_path}/imgs/{save_name}.png")
        
        plt.close()
        
        return fig
    

    
    def _format_hic_ticks(self, ax, matrix_size: int, genomic_start_mb: float, genomic_end_mb: float):
        """格式化Hi-C矩阵的基因组坐标刻度"""
        # 设置刻度位置（5个主要刻度）
        n_ticks = 5
        tick_positions = np.linspace(0, matrix_size-1, n_ticks)
        genomic_labels = np.linspace(genomic_start_mb, genomic_end_mb, n_ticks)
        genomic_labels = [f'{pos:.1f}' for pos in genomic_labels]
        
        # 设置x轴刻度
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(genomic_labels)
        ax.set_xlabel('Genomic Position (Mb)', fontsize=12)
        
        # 设置y轴刻度（对于Hi-C矩阵，y轴也是基因组位置）
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(genomic_labels)
    
    def _format_track_ticks(self, ax, matrix_size: int, genomic_start_mb: float, genomic_end_mb: float):
        """格式化1D track的基因组坐标刻度"""
        # 设置刻度位置（5个主要刻度）
        n_ticks = 5
        tick_positions = np.linspace(0, matrix_size-1, n_ticks)
        genomic_labels = np.linspace(genomic_start_mb, genomic_end_mb, n_ticks)
        genomic_labels = [f'{pos:.1f}' for pos in genomic_labels]
        
        # 只设置x轴刻度
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(genomic_labels)
    
    def _format_genomic_ticks(self, ax, matrix_size: int):
        """格式化基因组坐标刻度 - 保留向后兼容性"""
        # 设置x轴刻度
        tick_positions = np.linspace(0, matrix_size-1, 5)
        genomic_labels = np.linspace(self.start_pos, self.start_pos + 2097152, 5)
        genomic_labels = [f'{pos/1000000:.1f}' for pos in genomic_labels]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(genomic_labels)
        
        # 设置y轴刻度（对于Hi-C矩阵）
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(genomic_labels)
    
    def create_individual_tracks(self, 
                               prediction: np.ndarray,
                               chipseq_data: np.ndarray,
                               gradcam_1d: Optional[np.ndarray] = None,
                               gradcam_results: Optional[Dict] = None,
                               attention_weights: Optional[Dict] = None,
                               impact_scores: Optional[np.ndarray] = None,
                               perturb_starts: Optional[np.ndarray] = None):
        """创建单独的track可视化"""
        
        # 1. Hi-C预测单独图 - 正方形显示
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=2, aspect='equal', origin='upper')
        ax1.set_title(f'Hi-C Prediction - {self.chr_name}:{self.start_pos//1000000:.1f}Mb')
        
        # 使用新的刻度格式化方法
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        self._format_hic_ticks(ax1, prediction.shape[0], genomic_start_mb, genomic_end_mb)
        
        # 绘制图形以获取准确的位置信息
        fig1.canvas.draw()
        
        # 创建与Hi-C图高度匹配的colorbar
        pos1 = ax1.get_position()
        cbar_ax = fig1.add_axes([pos1.x1 + 0.02, pos1.y0, 0.02, pos1.height])
        cbar1 = fig1.colorbar(im1, cax=cbar_ax)
        cbar1.set_label('Contact Frequency', fontsize=10)
        
        plt.savefig(f'{self.save_path}/imgs/hic_prediction_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ChIP-seq track单独图
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        # 使用像素坐标以便与Hi-C对齐
        matrix_size = prediction.shape[0]
        pixel_positions = np.linspace(0, matrix_size-1, len(chipseq_data))
        ax2.plot(pixel_positions, chipseq_data, 'b-', linewidth=1)
        ax2.fill_between(pixel_positions, chipseq_data, alpha=0.3, color='blue')
        ax2.set_title('ChIP-seq Track')
        ax2.set_xlabel('Genomic Position (Mb)')
        ax2.set_ylabel('ChIP-seqSignal')
        ax2.set_xlim(0, matrix_size-1)
        self._format_track_ticks(ax2, matrix_size, genomic_start_mb, genomic_end_mb)
        ax2.grid(True, alpha=0.3)
        plt.savefig(f'{self.save_path}/imgs/chip_track_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Grad-CAM单独图
        if gradcam_1d is not None:
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            gradcam_pixel_positions = np.linspace(0, matrix_size-1, len(gradcam_1d))
            ax3.plot(gradcam_pixel_positions, gradcam_1d, 'g-', linewidth=1)
            ax3.fill_between(gradcam_pixel_positions, gradcam_1d, alpha=0.3, color='green')
            ax3.set_title('Grad-CAM Analysis (Main)')
            ax3.set_xlabel('Genomic Position (Mb)')
            ax3.set_ylabel('Grad-CAM Score')
            ax3.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax3, matrix_size, genomic_start_mb, genomic_end_mb)
            ax3.grid(True, alpha=0.3)
            plt.savefig(f'{self.save_path}/imgs/gradcam_individual.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3b. 多层Grad-CAM单独图
        if gradcam_results is not None:
            colors = ['green', 'cyan', 'blue', 'purple']
            for idx, (layer_name, gradcam_data) in enumerate(gradcam_results.items()):
                if gradcam_data is not None and len(gradcam_data) > 0:
                    fig3b, ax3b = plt.subplots(figsize=(12, 3))
                    gradcam_pixel_positions = np.linspace(0, matrix_size-1, len(gradcam_data))
                    
                    color = colors[idx % len(colors)]
                    ax3b.plot(gradcam_pixel_positions, gradcam_data, color=color, linewidth=1)
                    ax3b.fill_between(gradcam_pixel_positions, gradcam_data, alpha=0.3, color=color)
                    ax3b.set_title(f'Grad-CAM Analysis - {layer_name}')
                    ax3b.set_xlabel('Genomic Position (Mb)')
                    ax3b.set_ylabel('Grad-CAM Score')
                    ax3b.set_xlim(0, matrix_size-1)
                    self._format_track_ticks(ax3b, matrix_size, genomic_start_mb, genomic_end_mb)
                    ax3b.grid(True, alpha=0.3)
                    plt.savefig(f'{self.save_path}/imgs/gradcam_{layer_name}_individual.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 4. Attention权重单独图
        if attention_weights is not None:
            for attn_name, attn_data in attention_weights.items():
                if attn_data is not None:
                    if isinstance(attn_data, dict):
                        # 新的attention数据结构
                        # 为每种attention类型创建单独图
                        attention_plots = {}
                        
                        # 优先使用256长度的数据
                        if 'attention_1d_256' in attn_data:
                            attention_plots['1D_256'] = attn_data['attention_1d_256']
                        elif 'attention_1d' in attn_data:
                            attention_plots['1D'] = attn_data['attention_1d']
                        if 'regional_attention' in attn_data:
                            attention_plots['Regional'] = attn_data['regional_attention']
                        if 'global_averaged' in attn_data:
                            # 从2D生成1D用于可视化
                            attn_2d = attn_data['global_averaged']
                            attention_plots['Global_Avg'] = np.mean(attn_2d, axis=0)
                        
                        for plot_type, plot_data in attention_plots.items():
                            if plot_data is not None and len(plot_data) > 0:
                                fig4, ax4 = plt.subplots(figsize=(12, 3))
                                
                                # 处理像素位置
                                if len(plot_data) == matrix_size:
                                    attn_pixel_positions = np.arange(matrix_size)
                                else:
                                    attn_pixel_positions = np.linspace(0, matrix_size-1, len(plot_data))
                                
                                ax4.plot(attn_pixel_positions, plot_data, 'orange', linewidth=1)
                                ax4.fill_between(attn_pixel_positions, plot_data, alpha=0.3, color='orange')
                                ax4.set_title(f'Attention Weights - {attn_name} ({plot_type})')
                                ax4.set_xlabel('Genomic Position (Mb)')
                                ax4.set_ylabel('Attention Score')
                                ax4.set_xlim(0, matrix_size-1)
                                self._format_track_ticks(ax4, matrix_size, genomic_start_mb, genomic_end_mb)
                                ax4.grid(True, alpha=0.3)
                                plt.savefig(f'{self.save_path}/imgs/attention_{attn_name}_{plot_type}_individual.png', 
                                           dpi=300, bbox_inches='tight')
                                plt.close()
                        
                        # 如果有2D attention矩阵，也创建热图
                        if 'global_averaged' in attn_data or 'head_averaged' in attn_data:
                            attn_2d = attn_data.get('global_averaged')
                            if attn_2d is None:
                                attn_2d = attn_data.get('head_averaged')
                            if attn_2d is not None:
                                fig5, ax5 = plt.subplots(figsize=(10, 8))
                                im = ax5.imshow(attn_2d, cmap='Blues', aspect='auto', origin='upper')
                                ax5.set_title(f'Attention Matrix - {attn_name}')
                                ax5.set_xlabel('Target Position')
                                ax5.set_ylabel('Source Position')
                                plt.colorbar(im, ax=ax5, label='Attention Weight')
                                plt.savefig(f'{self.save_path}/imgs/attention_{attn_name}_matrix.png', 
                                           dpi=300, bbox_inches='tight')
                                plt.close()
                    
                    else:
                        # 兼容旧格式
                        fig4, ax4 = plt.subplots(figsize=(12, 3))
                        
                        if len(attn_data.shape) >= 2:
                            attn_1d = attn_data.mean(axis=0)
                        else:
                            attn_1d = attn_data
                        
                        attn_pixel_positions = np.linspace(0, matrix_size-1, len(attn_1d))
                        ax4.plot(attn_pixel_positions, attn_1d, 'orange', linewidth=1)
                        ax4.fill_between(attn_pixel_positions, attn_1d, alpha=0.3, color='orange')
                        ax4.set_title(f'Attention Weights - {attn_name}')
                        ax4.set_xlabel('Genomic Position (Mb)')
                        ax4.set_ylabel('Attention Score')
                        ax4.set_xlim(0, matrix_size-1)
                        self._format_track_ticks(ax4, matrix_size, genomic_start_mb, genomic_end_mb)
                        ax4.grid(True, alpha=0.3)
                        plt.savefig(f'{self.save_path}/imgs/attention_{attn_name}_individual.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
        
        # 5. Impact scores单独图
        if impact_scores is not None and perturb_starts is not None:
            fig5, ax5 = plt.subplots(figsize=(12, 4))
            
            # 转换为像素坐标
            genomic_range = 2097152  # 2MB in bp
            impact_pixel_positions = []
            for start_pos in perturb_starts:
                relative_pos = start_pos - self.start_pos
                pixel_pos = (relative_pos / genomic_range) * matrix_size
                impact_pixel_positions.append(pixel_pos)
            impact_pixel_positions = np.array(impact_pixel_positions)
            
            # 计算bar宽度（假设step_size为8192）
            pixel_width = (8192 / genomic_range) * matrix_size * 0.8
            
            ax5.bar(impact_pixel_positions, impact_scores, 
                   width=pixel_width, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax5.set_title('Impact Scores')
            ax5.set_xlabel('Genomic Position (Mb)')
            ax5.set_ylabel('Impact Score')
            ax5.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax5, matrix_size, genomic_start_mb, genomic_end_mb)
            ax5.grid(True, alpha=0.3)
            plt.savefig(f'{self.save_path}/imgs/impact_scores_individual.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("所有单独track图像已保存")
    
    def create_attention_heatmap(self, attention_weights: Dict):
        """创建attention权重的热图可视化"""
        if not attention_weights:
            return
        
        for attn_name, attn_data in attention_weights.items():
            if attn_data is not None:
                if isinstance(attn_data, dict):
                    # 新的attention数据结构
                    
                    # 1. 如果有原始attention权重，创建多层热图
                    if 'raw_attention' in attn_data:
                        raw_attn = attn_data['raw_attention']
                        
                        if len(raw_attn.shape) == 4:
                            # [layers, heads, seq_len, seq_len]
                            n_layers, n_heads, seq_len, _ = raw_attn.shape
                            
                            # 创建每层的热图
                            for layer_idx in range(min(n_layers, 4)):  # 最多显示4层
                                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                                axes = axes.flatten()
                                
                                for head_idx in range(min(n_heads, 8)):  # 最多显示8个head
                                    ax = axes[head_idx]
                                    attn_matrix = raw_attn[layer_idx, head_idx]
                                    
                                    im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', origin='upper')
                                    ax.set_title(f'Layer{layer_idx} Head{head_idx}', fontsize=10)
                                    ax.set_xlabel('Target Position')
                                    ax.set_ylabel('Source Position')
                                
                                # 隐藏多余的子图
                                for idx in range(n_heads, 8):
                                    axes[idx].set_visible(False)
                                
                                plt.tight_layout()
                                plt.savefig(f'{self.save_path}/imgs/attention_heatmap_{attn_name}_layer{layer_idx}.png', 
                                           dpi=300, bbox_inches='tight')
                                plt.close()
                        
                        elif len(raw_attn.shape) == 3:
                            # [heads, seq_len, seq_len]
                            n_heads, seq_len, _ = raw_attn.shape
                            
                            # 创建多head热图
                            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                            axes = axes.flatten()
                            
                            for head_idx in range(min(n_heads, 8)):
                                ax = axes[head_idx]
                                attn_matrix = raw_attn[head_idx]
                                
                                im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', origin='upper')
                                ax.set_title(f'Head {head_idx}', fontsize=10)
                                ax.set_xlabel('Target Position')
                                ax.set_ylabel('Source Position')
                            
                            # 隐藏多余的子图
                            for idx in range(n_heads, 8):
                                axes[idx].set_visible(False)
                            
                            plt.tight_layout()
                            plt.savefig(f'{self.save_path}/imgs/attention_heatmap_{attn_name}_heads.png', 
                                       dpi=300, bbox_inches='tight')
                            plt.close()
                    
                    # 2. 创建平均attention热图
                    heatmap_candidates = ['global_averaged', 'head_averaged', 'layer_averaged']
                    for candidate in heatmap_candidates:
                        if candidate in attn_data:
                            attn_matrix = attn_data[candidate]
                            
                            if len(attn_matrix.shape) == 3:
                                # 多层情况，为每层创建热图
                                for layer_idx in range(min(attn_matrix.shape[0], 4)):
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    im = ax.imshow(attn_matrix[layer_idx], cmap='Blues', aspect='auto', origin='upper')
                                    ax.set_title(f'Attention Matrix - {attn_name} ({candidate}) Layer {layer_idx}')
                                    ax.set_xlabel('Target Position')
                                    ax.set_ylabel('Source Position')
                                    plt.colorbar(im, ax=ax, label='Attention Weight')
                                    plt.savefig(f'{self.save_path}/imgs/attention_heatmap_{attn_name}_{candidate}_layer{layer_idx}.png', 
                                               dpi=300, bbox_inches='tight')
                                    plt.close()
                            
                            elif len(attn_matrix.shape) == 2:
                                # 单层情况
                                fig, ax = plt.subplots(figsize=(10, 8))
                                im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', origin='upper')
                                ax.set_title(f'Attention Matrix - {attn_name} ({candidate})')
                                ax.set_xlabel('Target Position')
                                ax.set_ylabel('Source Position')
                                plt.colorbar(im, ax=ax, label='Attention Weight')
                                plt.savefig(f'{self.save_path}/imgs/attention_heatmap_{attn_name}_{candidate}.png', 
                                           dpi=300, bbox_inches='tight')
                                plt.close()
                            
                            break  # 只创建第一个找到的热图
                
                else:
                    # 兼容旧格式
                    if len(attn_data.shape) >= 2:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # 如果是3D数组，选择第一个batch
                        if len(attn_data.shape) == 3:
                            heatmap_data = attn_data[0]
                        else:
                            heatmap_data = attn_data
                        
                        im = ax.imshow(heatmap_data, cmap='Blues', aspect='auto')
                        ax.set_title(f'Attention Heatmap - {attn_name}')
                        ax.set_xlabel('Position')
                        ax.set_ylabel('Head/Layer')
                        plt.colorbar(im, ax=ax)
                        
                        plt.savefig(f'{self.save_path}/imgs/attention_heatmap_{attn_name}.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
        
        print("Attention热图已保存")

    def create_paper_style_visualization_gram(self, 
                                             prediction: np.ndarray,
                                             chipseq_data: np.ndarray,
                                             gram_1d: Optional[np.ndarray] = None,
                                             impact_scores: Optional[np.ndarray] = None,
                                             perturb_starts: Optional[np.ndarray] = None,
                                             step_size: int = 8192):
        """创建GRAM版本的论文样式综合可视化"""
        
        # 计算track数量
        n_tracks = 2  # Hi-C + ChIP-seq
        if gram_1d is not None:
            n_tracks += 1
        if impact_scores is not None:
            n_tracks += 1
        
        # 创建图形，使用gridspec来精确控制布局
        fig = plt.figure(figsize=(12, 12))
        
        # 设置基因组坐标范围（Mb）
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        
        # 先创建Hi-C图，然后获取其精确位置用于对齐其他track
        matrix_size = prediction.shape[0]
        
        # 1. Hi-C预测热图 (顶部) - 创建临时GridSpec
        temp_gs = gridspec.GridSpec(1, 1, top=0.9, bottom=0.6)
        ax1 = fig.add_subplot(temp_gs[0])
        
        # 使用正方形比例显示Hi-C矩阵
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=2, 
                        aspect='equal', interpolation='bilinear', origin='upper')
        
        # 设置标题和标签
        ax1.set_title(f'{self.chr_name}: {genomic_start_mb:.1f}-{genomic_end_mb:.1f} Mb', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Genomic distance (Mb)', fontsize=12)
        
        # 设置Hi-C的刻度，对应基因组位置
        self._format_hic_ticks(ax1, matrix_size, genomic_start_mb, genomic_end_mb)
        
        # 绘制图形以获取Hi-C图的精确位置
        fig.canvas.draw()
        hic_pos = ax1.get_position()
        
        # 存储Hi-C图的位置信息
        hic_left = hic_pos.x0
        hic_right = hic_pos.x1  
        hic_width = hic_pos.width
        hic_bottom = hic_pos.y0
        
        track_idx = 1
        
        # 2. ChIP-seq track - 手动设置位置与Hi-C对齐
        track_height = 0.08
        track_spacing = 0.02
        current_y = hic_bottom - track_spacing - track_height
        
        ax2 = fig.add_axes([hic_left, current_y, hic_width, track_height])
        # 使用像素坐标与Hi-C对齐
        pixel_positions = np.linspace(0, matrix_size-1, len(chipseq_data))
        ax2.fill_between(pixel_positions, chipseq_data, alpha=0.7, color='blue', linewidth=0)
        ax2.plot(pixel_positions, chipseq_data, 'b-', linewidth=0.8)
        
        ax2.set_ylabel('ChIP-seq', fontsize=12)
        ax2.set_xlim(0, matrix_size-1)
        # 设置与Hi-C相同的x轴刻度
        self._format_track_ticks(ax2, matrix_size, genomic_start_mb, genomic_end_mb)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        current_y -= (track_height + track_spacing)
        track_idx += 1
        
        # 3. GRAM track
        if gram_1d is not None:
            ax_gram = fig.add_axes([hic_left, current_y, hic_width, track_height])
            # 使用原始GRAM值，不进行标准化
            # 使用像素坐标与Hi-C对齐
            gram_pixel_positions = np.linspace(0, matrix_size-1, len(gram_1d))
            ax_gram.plot(gram_pixel_positions, gram_1d, color='green', linewidth=1.2, alpha=0.9)
            
            ax_gram.set_ylabel('GRAM', fontsize=12)
            ax_gram.set_xlim(0, matrix_size-1)
            # 统一设置GRAM track的Y轴范围为0-10
            ax_gram.set_ylim(0, 10)
            self._format_track_ticks(ax_gram, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_gram.grid(True, alpha=0.3)
            ax_gram.spines['top'].set_visible(False)
            ax_gram.spines['right'].set_visible(False)
            
            current_y -= (track_height + track_spacing)
            track_idx += 1
        
        # 4. Impact scores track
        if impact_scores is not None:
            ax_impact = fig.add_axes([hic_left, current_y, hic_width, track_height])
            # 使用像素坐标与Hi-C对齐
            impact_pixel_positions = np.linspace(0, matrix_size-1, len(impact_scores))
            ax_impact.plot(impact_pixel_positions, impact_scores, 'r-', linewidth=0.8)
            
            ax_impact.set_ylabel('Impact', fontsize=12)
            ax_impact.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax_impact, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_impact.grid(True, alpha=0.3)
            ax_impact.spines['top'].set_visible(False)
            ax_impact.spines['right'].set_visible(False)
        
        # 添加colorbar
        pos1 = ax1.get_position()
        cbar_ax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.02, pos1.height])
        cbar1 = fig.colorbar(im1, cax=cbar_ax)
        cbar1.set_label('Contact Frequency', fontsize=10)
        
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_paper_style_gram.png', 
                    dpi=300, bbox_inches='tight')
        print(f"GRAM版本论文样式综合图保存为: {self.save_path}/imgs/{self.chr_name}_{self.start_pos}_paper_style_gram.png")
        plt.close()

    def create_individual_tracks_gram(self, 
                                     prediction: np.ndarray,
                                     chipseq_data: np.ndarray,
                                     gram_1d: Optional[np.ndarray] = None,
                                     alpha_weights: Optional[np.ndarray] = None,
                                     impact_scores: Optional[np.ndarray] = None,
                                     perturb_starts: Optional[np.ndarray] = None):
        """创建GRAM版本的单独track可视化"""
        
        # 计算基因组坐标范围（Mb）
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        matrix_size = prediction.shape[0]
        
        # 1. Hi-C预测单独图 - 正方形显示
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=2, aspect='equal', origin='upper')
        ax1.set_title(f'Hi-C Prediction - {self.chr_name}:{self.start_pos//1000000:.1f}Mb')
        
        # 使用新的刻度格式化方法
        self._format_hic_ticks(ax1, matrix_size, genomic_start_mb, genomic_end_mb)
        
        # 绘制图形以获取准确的位置信息
        fig1.canvas.draw()
        
        # 创建与Hi-C图高度匹配的colorbar
        pos1 = ax1.get_position()
        cbar_ax = fig1.add_axes([pos1.x1 + 0.02, pos1.y0, 0.02, pos1.height])
        cbar1 = fig1.colorbar(im1, cax=cbar_ax)
        cbar1.set_label('Contact Frequency', fontsize=10)
        
        plt.savefig(f'{self.save_path}/imgs/hic_prediction_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ChIP-seq track单独图
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        # 使用像素坐标以便与Hi-C对齐
        pixel_positions = np.linspace(0, matrix_size-1, len(chipseq_data))
        ax2.plot(pixel_positions, chipseq_data, 'b-', linewidth=1)
        ax2.fill_between(pixel_positions, chipseq_data, alpha=0.3, color='blue')
        ax2.set_title('ChIP-seq Track')
        ax2.set_xlabel('Genomic Position (Mb)')
        ax2.set_ylabel('ChIP-seq Signal')
        ax2.set_xlim(0, matrix_size-1)
        self._format_track_ticks(ax2, matrix_size, genomic_start_mb, genomic_end_mb)
        ax2.grid(True, alpha=0.3)
        plt.savefig(f'{self.save_path}/imgs/chip_track_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. GRAM track单独图
        if gram_1d is not None:
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            # 使用原始GRAM值，不进行标准化
            gram_pixel_positions = np.linspace(0, matrix_size-1, len(gram_1d))
            ax3.plot(gram_pixel_positions, gram_1d, 'g-', linewidth=1.5, alpha=0.9)
            ax3.set_title('GRAM Analysis')
            ax3.set_xlabel('Genomic Position (Mb)')
            ax3.set_ylabel('GRAM Score')
            ax3.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax3, matrix_size, genomic_start_mb, genomic_end_mb)
            ax3.grid(True, alpha=0.3)
            plt.savefig(f'{self.save_path}/imgs/gram_track_individual.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Alpha权重单独图
        if alpha_weights is not None:
            fig4, ax4 = plt.subplots(figsize=(12, 3))
            
            # 分别绘制左右两边的折线，使用不同颜色
            # 左边：0-127 (蓝色)
            left_indices = np.arange(128)
            ax4.plot(left_indices, alpha_weights[:128], 'b-', linewidth=1, label='Left (0-127)')
            
            # 右边：128-255 (绿色)
            right_indices = np.arange(128, len(alpha_weights))
            ax4.plot(right_indices, alpha_weights[128:], 'g-', linewidth=1, label='Right (128-255)')
            
            ax4.set_title('Alpha Weights')
            ax4.set_xlabel('Channel')
            ax4.set_ylabel('Weight')
            # 设置x轴范围，去掉左右空白
            ax4.set_xlim(0, len(alpha_weights)-1)
            # 在128位置添加虚线分割
            ax4.axvline(x=128, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            plt.savefig(f'{self.save_path}/imgs/alpha_weights_individual.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Impact scores单独图
        if impact_scores is not None:
            fig5, ax5 = plt.subplots(figsize=(12, 3))
            impact_pixel_positions = np.linspace(0, matrix_size-1, len(impact_scores))
            ax5.plot(impact_pixel_positions, impact_scores, 'r-', linewidth=1)
            ax5.set_title('Impact Scores')
            ax5.set_xlabel('Genomic Position (Mb)')
            ax5.set_ylabel('Impact Score')
            ax5.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax5, matrix_size, genomic_start_mb, genomic_end_mb)
            ax5.grid(True, alpha=0.3)
            plt.savefig(f'{self.save_path}/imgs/impact_scores_individual.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("GRAM版本所有单独track图像已保存") 

    def create_multi_layer_gram_comparison(self, 
                                         prediction: np.ndarray,
                                         chipseq_data: np.ndarray,
                                         gram_results: Dict[str, np.ndarray],
                                         impact_scores: Optional[np.ndarray] = None,
                                         perturb_starts: Optional[np.ndarray] = None,
                                         step_size: int = 8192):
        """创建多layer GRAM比较可视化"""
        
        # 计算track数量
        n_tracks = 2  # Hi-C + ChIP-seq
        n_gram_layers = len(gram_results)
        n_tracks += n_gram_layers
        if impact_scores is not None:
            n_tracks += 1
        
        # 创建图形
        fig = plt.figure(figsize=(14, 3 * n_tracks))
        
        # 设置基因组坐标范围（Mb）
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        matrix_size = prediction.shape[0]
        
        # 1. Hi-C预测热图 (顶部)
        ax1 = plt.subplot(n_tracks, 1, 1)
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=2, 
                        aspect='equal', interpolation='bilinear', origin='upper')
        
        ax1.set_title(f'{self.chr_name}: {genomic_start_mb:.1f}-{genomic_end_mb:.1f} Mb', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Genomic distance (Mb)', fontsize=12)
        self._format_hic_ticks(ax1, matrix_size, genomic_start_mb, genomic_end_mb)
        
        # 2. ChIP-seq track
        ax2 = plt.subplot(n_tracks, 1, 2)
        pixel_positions = np.linspace(0, matrix_size-1, len(chipseq_data))
        ax2.fill_between(pixel_positions, chipseq_data, alpha=0.7, color='blue', linewidth=0)
        ax2.plot(pixel_positions, chipseq_data, 'b-', linewidth=0.8)
        ax2.set_ylabel('ChIP-seq', fontsize=12)
        ax2.set_xlim(0, matrix_size-1)
        self._format_track_ticks(ax2, matrix_size, genomic_start_mb, genomic_end_mb)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. 多个GRAM tracks
        colors = ['green', 'red', 'purple', 'orange', 'brown']
        for i, (layer_name, gram_1d) in enumerate(gram_results.items()):
            ax_gram = plt.subplot(n_tracks, 1, 3 + i)
            gram_pixel_positions = np.linspace(0, matrix_size-1, len(gram_1d))
            color = colors[i % len(colors)]

            gram_norm = self._normalize_track(gram_1d)
            ax_gram.plot(gram_pixel_positions, gram_norm, color=color, linewidth=1.2, alpha=0.9)
            ax_gram.set_ylabel(f'GRAM(norm)\n({layer_name.split(".")[-1]})', fontsize=12)
            ax_gram.set_xlim(0, matrix_size-1)
            ax_gram.set_ylim(0, 1)
            self._format_track_ticks(ax_gram, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_gram.grid(True, alpha=0.3)
            ax_gram.spines['top'].set_visible(False)
            ax_gram.spines['right'].set_visible(False)
        
        # 4. Impact scores track (如果有)
        if impact_scores is not None:
            ax_impact = plt.subplot(n_tracks, 1, 3 + n_gram_layers)
            impact_pixel_positions = np.linspace(0, matrix_size-1, len(impact_scores))
            ax_impact.plot(impact_pixel_positions, impact_scores, 'r-', linewidth=0.8)
            ax_impact.set_ylabel('Impact', fontsize=12)
            ax_impact.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax_impact, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_impact.grid(True, alpha=0.3)
            ax_impact.spines['top'].set_visible(False)
            ax_impact.spines['right'].set_visible(False)
        
        # 添加colorbar
        pos1 = ax1.get_position()
        cbar_ax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.02, pos1.height])
        cbar1 = fig.colorbar(im1, cax=cbar_ax)
        cbar1.set_label('Contact Frequency', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/multi_layer_gram_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.save_path}/multi_layer_gram_comparison.pdf',
                    bbox_inches='tight')
        print(f"多layer GRAM比较图保存为: {self.save_path}/multi_layer_gram_comparison.png")
        plt.close() 

    def create_multi_layer_gram_paper_style(self, 
                                           prediction: np.ndarray,
                                           chipseq_data: np.ndarray,
                                           gram_results: Dict[str, np.ndarray],
                                           impact_scores: Optional[np.ndarray] = None,
                                           perturb_starts: Optional[np.ndarray] = None,
                                           step_size: int = 8192):
        """创建多layer GRAM的paper style综合可视化 - 完全匹配参考图像样式"""
        
        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 计算track数量
        n_gram_layers = len(gram_results)
        n_tracks = 2 + n_gram_layers  # Hi-C + ChIP-seq + GRAM layers
        if impact_scores is not None:
            n_tracks += 1
        
        # 设置基因组坐标范围（Mb）
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        matrix_size = prediction.shape[0]
        
        # 创建figure - 使用与参考图像相同的比例
        fig_width = 12
        heatmap_height = 4.5  # 热图高度
        track_height = 1.2    # 每个track高度  
        fig_height = heatmap_height + n_tracks * track_height * 0.6
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # 计算gridspec布局 - 确保完美对齐
        from matplotlib.gridspec import GridSpec
        
        # 创建gridspec：热图占更多行，每个track占1行
        height_ratios = [4] + [1] * (n_tracks - 1)  # 热图高度是track的4倍
        gs = GridSpec(n_tracks, 1, height_ratios=height_ratios, hspace=0.05)
        
        # 1. Hi-C预测热图 (顶部)
        ax1 = fig.add_subplot(gs[0])
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=2, 
                        aspect='equal', interpolation='bilinear', origin='upper')
        
        ax1.set_title(f'{self.chr_name}: {genomic_start_mb:.1f}-{genomic_end_mb:.1f} Mb', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Genomic distance (Mb)', fontsize=12)
        
        # 设置热图刻度
        ax1.set_xlim(-0.5, matrix_size-0.5)
        ax1.set_ylim(matrix_size-0.5, -0.5)  # 反转Y轴使其从上到下
        
        # Y轴刻度 - 基因组距离
        y_positions = np.linspace(0, matrix_size-1, 5)
        y_labels = [f"{genomic_start_mb:.1f}", f"{genomic_start_mb + 0.25*(genomic_end_mb-genomic_start_mb):.1f}",
                   f"{genomic_start_mb + 0.5*(genomic_end_mb-genomic_start_mb):.1f}", 
                   f"{genomic_start_mb + 0.75*(genomic_end_mb-genomic_start_mb):.1f}", f"{genomic_end_mb:.1f}"]
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(y_labels)
        
        # 隐藏热图的X轴刻度
        ax1.set_xticks([])
        
        # 2. ChIP-seq track
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        # 重新采样ChIP-seq数据到matrix_size点
        chipseq_resampled = np.interp(np.linspace(0, len(chipseq_data)-1, matrix_size), 
                                     np.arange(len(chipseq_data)), chipseq_data)
        x_positions = np.arange(matrix_size)
        
        ax2.fill_between(x_positions, chipseq_resampled, alpha=0.7, color='blue', linewidth=0)
        ax2.plot(x_positions, chipseq_resampled, 'b-', linewidth=0.8)
        ax2.set_ylabel('ChIP-seq', fontsize=12)
        ax2.set_xlim(-0.5, matrix_size-0.5)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xticks([])  # 隐藏X轴刻度
        
        # 3. 多个GRAM tracks
        colors = ['green', 'red', 'purple']
        layer_names_short = {
            'encoder.gate_fusion': 'GRAM_fusion',
            'encoder.encoder_seq': 'GRAM_seq', 
            'encoder.encoder_epi': 'GRAM_epi'
        }
        
        track_idx = 2
        for i, (layer_name, gram_1d) in enumerate(gram_results.items()):
            ax_gram = fig.add_subplot(gs[track_idx], sharex=ax1)
            # 确保GRAM数据与matrix_size对齐
            if len(gram_1d) != matrix_size:
                gram_resampled = np.interp(np.linspace(0, len(gram_1d)-1, matrix_size), 
                                         np.arange(len(gram_1d)), gram_1d)
            else:
                gram_resampled = gram_1d
                
            color = colors[i % len(colors)]
            short_name = layer_names_short.get(layer_name, layer_name.split('.')[-1])
            
            ax_gram.plot(x_positions, gram_resampled, color=color, linewidth=1.2, alpha=0.9)
            ax_gram.set_ylabel(short_name, fontsize=12)
            ax_gram.set_xlim(-0.5, matrix_size-0.5)
            ax_gram.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax_gram.spines['top'].set_visible(False)
            ax_gram.spines['right'].set_visible(False)
            ax_gram.set_xticks([])  # 暂时隐藏X轴刻度
            track_idx += 1
        
        # 4. Impact scores track (如果有)
        if impact_scores is not None:
            ax_impact = fig.add_subplot(gs[track_idx], sharex=ax1)
            # 重新采样impact scores到matrix_size点
            if len(impact_scores) != matrix_size:
                impact_resampled = np.interp(np.linspace(0, len(impact_scores)-1, matrix_size), 
                                           np.arange(len(impact_scores)), impact_scores)
            else:
                impact_resampled = impact_scores
                
            ax_impact.plot(x_positions, impact_resampled, 'r-', linewidth=0.8)
            ax_impact.set_ylabel('Impact', fontsize=12)
            ax_impact.set_xlim(-0.5, matrix_size-0.5)
            ax_impact.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax_impact.spines['top'].set_visible(False)
            ax_impact.spines['right'].set_visible(False)
            track_idx += 1
        
        # 在最后一个subplot上显示X轴刻度
        last_ax = fig.get_axes()[-1]
        x_tick_positions = np.linspace(0, matrix_size-1, 6)
        x_labels = [f"{genomic_start_mb + (i/(matrix_size-1)) * (genomic_end_mb - genomic_start_mb):.1f}" 
                   for i in x_tick_positions]
        last_ax.set_xticks(x_tick_positions)
        last_ax.set_xticklabels(x_labels)
        last_ax.set_xlabel('Genomic Position (Mb)', fontsize=12)
        
        # 添加colorbar - 与热图对齐
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im1, cax=cax)
        cbar.set_label('Contact Frequency', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'{self.save_path}/multi_layer_gram_paper_style.png', 
                    dpi=300, bbox_inches='tight')
        print(f"多layer GRAM paper style图保存为: {self.save_path}/multi_layer_gram_paper_style.png")
        plt.close() 

    def create_paper_style_visualization_multi_gram(self, 
                                                   prediction: np.ndarray,
                                                   chipseq_data: np.ndarray,
                                                   gram_results: Dict[str, np.ndarray],
                                                   impact_scores: Optional[np.ndarray] = None,
                                                   perturb_starts: Optional[np.ndarray] = None,
                                                   step_size: int = 8192):
        """创建多GRAM版本的论文样式综合可视化 - 基于现有的单GRAM函数"""
        
        # 计算track数量
        n_tracks = 2  # Hi-C + ChIP-seq
        n_gram_layers = len(gram_results)
        n_tracks += n_gram_layers
        if impact_scores is not None:
            n_tracks += 1
        
        # 创建图形，使用gridspec来精确控制布局
        fig = plt.figure(figsize=(12, 12))
        
        # 设置基因组坐标范围（Mb）
        genomic_start_mb = self.start_pos / 1000000
        genomic_end_mb = (self.start_pos + 2097152) / 1000000
        
        # 先创建Hi-C图，然后获取其精确位置用于对齐其他track
        matrix_size = prediction.shape[0]
        
        # 1. Hi-C预测热图 (顶部) - 创建临时GridSpec
        temp_gs = gridspec.GridSpec(1, 1, top=0.9, bottom=0.6)
        ax1 = fig.add_subplot(temp_gs[0])
        
        # 使用正方形比例显示Hi-C矩阵
        hic_colormap = self.get_colormap_hic()
        im1 = ax1.imshow(prediction, cmap=hic_colormap, vmin=0, vmax=3, 
                        aspect='equal', interpolation='bilinear', origin='upper')
        
        # 设置标题和标签
        ax1.set_title(f'{self.chr_name}: {genomic_start_mb:.1f}-{genomic_end_mb:.1f} Mb', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Genomic distance (Mb)', fontsize=12)
        
        # 设置Hi-C的刻度，对应基因组位置
        self._format_hic_ticks(ax1, matrix_size, genomic_start_mb, genomic_end_mb)
        
        # 绘制图形以获取Hi-C图的精确位置
        fig.canvas.draw()
        hic_pos = ax1.get_position()
        
        # 存储Hi-C图的位置信息
        hic_left = hic_pos.x0
        hic_right = hic_pos.x1  
        hic_width = hic_pos.width
        hic_bottom = hic_pos.y0
        
        track_idx = 1
        
        # 2. ChIP-seq track - 手动设置位置与Hi-C对齐
        track_height = 0.08
        track_spacing = 0.02
        current_y = hic_bottom - track_spacing - track_height
        
        ax2 = fig.add_axes([hic_left, current_y, hic_width, track_height])
        # 使用像素坐标与Hi-C对齐
        pixel_positions = np.linspace(0, matrix_size-1, len(chipseq_data))
        ax2.fill_between(pixel_positions, chipseq_data, alpha=0.7, color='blue', linewidth=0)
        ax2.plot(pixel_positions, chipseq_data, 'b-', linewidth=0.8)
        
        ax2.set_ylabel('ChIP-seq', fontsize=12)
        ax2.set_xlim(0, matrix_size-1)
        # 设置与Hi-C相同的x轴刻度
        self._format_track_ticks(ax2, matrix_size, genomic_start_mb, genomic_end_mb)
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        current_y -= (track_height + track_spacing)
        track_idx += 1
        
        # 3. 多个GRAM tracks - 使用与原函数相同的布局方式
        colors = ['green', 'red', 'purple', 'orange', 'brown']
        layer_names_short = {
            'encoder.gate_fusion': 'GRAM_fusion',
            'encoder.encoder_seq': 'GRAM_seq', 
            'encoder.encoder_epi': 'GRAM_epi'
        }
        
        for i, (layer_name, gram_1d) in enumerate(gram_results.items()):
            ax_gram = fig.add_axes([hic_left, current_y, hic_width, track_height])
            gram_pixel_positions = np.linspace(0, matrix_size-1, len(gram_1d))
            color = colors[i % len(colors)]
            short_name = layer_names_short.get(layer_name, layer_name.split('.')[-1])

            gram_norm = self._normalize_track(gram_1d)
            ax_gram.plot(gram_pixel_positions, gram_norm, color=color, linewidth=1.2, alpha=0.9)

            ax_gram.set_ylabel(f'{short_name}\n(norm)', fontsize=12)
            ax_gram.set_xlim(0, matrix_size-1)
            ax_gram.set_ylim(0, 1)
            self._format_track_ticks(ax_gram, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_gram.grid(True, alpha=0.3)
            ax_gram.spines['top'].set_visible(False)
            ax_gram.spines['right'].set_visible(False)
            
            current_y -= (track_height + track_spacing)
            track_idx += 1
        
        # 4. Impact scores track
        if impact_scores is not None:
            ax_impact = fig.add_axes([hic_left, current_y, hic_width, track_height])
            # 使用像素坐标与Hi-C对齐
            impact_pixel_positions = np.linspace(0, matrix_size-1, len(impact_scores))
            ax_impact.plot(impact_pixel_positions, impact_scores, 'r-', linewidth=0.8)
            
            ax_impact.set_ylabel('Impact', fontsize=12)
            ax_impact.set_xlim(0, matrix_size-1)
            self._format_track_ticks(ax_impact, matrix_size, genomic_start_mb, genomic_end_mb)
            ax_impact.grid(True, alpha=0.3)
            ax_impact.spines['top'].set_visible(False)
            ax_impact.spines['right'].set_visible(False)
        
        # 添加colorbar - 与原函数完全相同
        pos1 = ax1.get_position()
        cbar_ax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.02, pos1.height])
        cbar1 = fig.colorbar(im1, cax=cbar_ax)
        cbar1.set_label('Contact Frequency', fontsize=10)
        
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_paper_style_multi_gram.png', 
                    dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_paper_style_multi_gram.pdf',
                    bbox_inches='tight')
        print(f"多GRAM版本论文样式综合图保存为: {self.save_path}/imgs/{self.chr_name}_{self.start_pos}_paper_style_multi_gram.png")
        plt.close() 
