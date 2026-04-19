import os
import numpy as np
import pandas as pd

class MatrixPlot:

    def __init__(self, output_path, image, prefix, celltype, chr_name, start_pos):
        self.output_path = output_path,
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos

        self.create_save_path(output_path, celltype, prefix)
        self.image = self.preprocess_image(image)

    def get_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        return color_map

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok = True)
        os.makedirs(f'{self.save_path}/npy', exist_ok = True)

    def preprocess_image(self, image):
        return image

    def plot(self, vmin = 0, vmax = 5):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (5, 5))
        color_map = self.get_colormap()
        ax.imshow(self.image, cmap = color_map, vmin = vmin, vmax = vmax)
        self.reformat_ticks(plt)
        return 

    def reformat_ticks(self, plt):
        # Rescale tick labels
        current_ticks = np.arange(0, 250, 50) / 0.8192
        plt.xticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        plt.yticks(current_ticks, self.rescale_coordinates(current_ticks, self.start_pos))
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        plt.xlabel(f'Chr{self.chr_name.replace("chr", "")}: {self.start_pos} - {self.start_pos + 2097152} ')#2097152
        self.save_data(plt)

    def rescale_coordinates(self, coords, zero_position):
        scaling_ratio = 8192
        replaced_coords = coords * scaling_ratio + zero_position
        coords_mb = replaced_coords / 1000000
        str_list = [f'{item:.2f}' for item in coords_mb]
        return str_list

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}.png', bbox_inches = 'tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}', self.image)

class MatrixPlotDeletion(MatrixPlot):
    def __init__(self, output_path, image, prefix, celltype, chr_name, start_pos, deletion_start, deletion_width, padding_type, show_deletion_line = False, insulation_scores = None):
        super().__init__(output_path, image, prefix, celltype, chr_name, start_pos)
        self.deletion_start = deletion_start
        self.deletion_width = deletion_width
        self.show_deletion_line = show_deletion_line
        self.padding_type = padding_type
        self.insulation_scores = insulation_scores

    def plot(self, vmin = 0, vmax = 5):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        # 如果有 insulation scores，则使用双面板布局
        if self.insulation_scores is not None:
            fig = plt.figure(figsize=(5, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.05)
            
            # 上方：Hi-C heatmap
            ax_heatmap = fig.add_subplot(gs[0])
            color_map = self.get_colormap()
            ax_heatmap.imshow(self.image, cmap=color_map, vmin=vmin, vmax=vmax)
            
            # 添加删除线（如果需要）
            if self.show_deletion_line:
                breakpoint_start = (self.deletion_start - self.start_pos) / 10000
                breakpoint_locus = breakpoint_start / 0.8192
                ax_heatmap.axvline(x=breakpoint_locus, color='black', alpha=0.5)
                ax_heatmap.axhline(y=breakpoint_locus, color='black', alpha=0.5)
            
            # 下方：Insulation Score track
            ax_track = fig.add_subplot(gs[1], sharex=ax_heatmap)
            x = np.arange(len(self.insulation_scores))
            ax_track.fill_between(x, self.insulation_scores, alpha=0.7, color='steelblue')
            ax_track.plot(x, self.insulation_scores, color='steelblue', linewidth=0.5)
            ax_track.set_ylabel('Ins. Score', fontsize=8)
            ax_track.set_xlim(0, len(self.insulation_scores) - 1)
            ax_track.spines['top'].set_visible(False)
            ax_track.spines['right'].set_visible(False)
            
            # 隐藏 heatmap 的 x 轴刻度（由 track 共享）
            plt.setp(ax_heatmap.get_xticklabels(), visible=False)
            
            self.reformat_ticks_with_track(plt, ax_heatmap, ax_track)
        else:
            # 无 insulation scores 时使用原有布局
            fig, ax = plt.subplots(figsize=(5, 5))
            color_map = self.get_colormap()
            ax.imshow(self.image, cmap=color_map, vmin=vmin, vmax=vmax)
            self.reformat_ticks(plt)
        return

    def reformat_ticks_with_track(self, plt, ax_heatmap, ax_track):
        """带 insulation score track 的坐标轴格式化"""
        breakpoint_start = (self.deletion_start - self.start_pos) / 10000
        breakpoint_end = (self.deletion_start - self.start_pos + self.deletion_width) / 10000
        total_window_size = (self.deletion_width + 2097152) / 10000
        
        before_ticks = np.arange(0, breakpoint_start - 50, 50) / 0.8192
        after_ticks = (np.arange((breakpoint_end // 50 + 2) * 50, total_window_size, 50) - self.deletion_width / 10000) / 0.8192
        breakpoint_locus = breakpoint_start / 0.8192
        
        current_ticks = np.append(before_ticks, after_ticks)
        current_ticks = np.append(current_ticks, breakpoint_start / 0.8192)
        
        display_ticks = np.append(before_ticks, after_ticks + self.deletion_width / 10000 / 0.8192)
        display_ticks = np.append(display_ticks, breakpoint_start / 0.8192)
        
        ticks_label = self.rescale_coordinates(display_ticks, self.start_pos)
        
        # heatmap y 轴
        ax_heatmap.set_yticks(current_ticks)
        ax_heatmap.set_yticklabels(ticks_label)
        ax_heatmap.set_ylabel('Genomic position (Mb)')
        
        # track x 轴
        ticks_label_x = list(ticks_label)
        ticks_label_x[-1] = f"{(self.deletion_start / 1000000):.2f}({(self.deletion_start + self.deletion_width) / 1000000:.2f})"
        ax_track.set_xticks(current_ticks)
        ax_track.set_xticklabels(ticks_label_x, fontsize=7)
        
        end_pos = self.start_pos + 2097152 + self.deletion_width
        ax_track.set_xlabel(f'Chr{self.chr_name.replace("chr", "")}: {self.start_pos} - {self.deletion_start} and {self.deletion_start + self.deletion_width} - {end_pos}', fontsize=8)
        
        self.save_data(plt)

    def reformat_ticks(self, plt):
        # Rescale tick labels
        breakpoint_start = (self.deletion_start - self.start_pos) / 10000 
        breakpoint_end = (self.deletion_start - self.start_pos + self.deletion_width) / 10000 
        # Used for generating ticks until the end of the window
        total_window_size = (self.deletion_width + 2097152 ) / 10000 #2097152
        # Generate ticks before and after breakpoint
        before_ticks = np.arange(0, breakpoint_start - 50, 50) / 0.8192
        after_ticks = (np.arange((breakpoint_end // 50 + 2) * 50, total_window_size, 50) - self.deletion_width / 10000) / 0.8192
        breakpoint_locus = breakpoint_start / 0.8192
        # Actual coordinates for each tick
        current_ticks = np.append(before_ticks, after_ticks)
        current_ticks = np.append(current_ticks, breakpoint_start / 0.8192)
        # Genomic coordinates used for display location after deletion
        display_ticks = np.append(before_ticks, after_ticks + self.deletion_width / 10000 / 0.8192)
        display_ticks = np.append(display_ticks, breakpoint_start / 0.8192)
        if self.show_deletion_line:
            plt.axline((breakpoint_locus, 0), (breakpoint_locus, 209), c = 'black', alpha = 0.5)#209
            plt.axline((0, breakpoint_locus), (209, breakpoint_locus), c = 'black', alpha = 0.5)#209
        # Generate tick label text
        ticks_label = self.rescale_coordinates(display_ticks, self.start_pos)
        plt.yticks(current_ticks, ticks_label)
        ticks_label[-1] = f"{(self.deletion_start / 1000000):.2f}({(self.deletion_start + self.deletion_width) / 1000000:.2f})"
        plt.xticks(current_ticks, ticks_label)
        # Format labels
        plt.ylabel('Genomic position (Mb)')
        end_pos = self.start_pos + 2097152 + self.deletion_width #2097152
        plt.xlabel(f'Chr{self.chr_name.replace("chr", "")}: {self.start_pos} - {self.deletion_start} and {self.deletion_start + self.deletion_width} - {end_pos} ')
        self.save_data(plt)

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}.png', bbox_inches = 'tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}', self.image)

class MatrixPlotPointScreen(MatrixPlotDeletion):

    def plot(self, vmin = -1, vmax = 1):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize = (5, 5))
        ax.imshow(self.image, cmap = 'RdBu_r', vmin = vmin, vmax = vmax)
        self.reformat_ticks(plt)
        return 

    def save_data(self, plt):
        plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}_diff.png', bbox_inches = 'tight')
        plt.close()
        np.save(f'{self.save_path}/npy/{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_padding_{self.padding_type}_diff', self.image)

class MatrixPlotScreen(MatrixPlot):
    def __init__(self, output_path, perturb_starts, perturb_ends, impact_score, tensor_diff, tensor_pred, tensor_deletion, prefix, celltype, chr_name, screen_start, screen_end, perturb_width, step_size, plot_impact_score):
        super().__init__(output_path, impact_score, prefix, celltype, chr_name, start_pos = None)
        self.perturb_starts = perturb_starts
        self.perturb_ends = perturb_ends
        self.impact_score = impact_score
        self.tensor_diff = tensor_diff
        self.tensor_pred = tensor_pred
        self.tensor_deletion = tensor_deletion
        self.screen_start = screen_start
        self.screen_end = screen_end
        self.perturb_width = perturb_width
        self.step_size = step_size
        self.plot_impact_score = plot_impact_score

    def create_save_path(self, output_path, celltype, prefix):
        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok = True)
        os.makedirs(f'{self.save_path}/npy', exist_ok = True)
        os.makedirs(f'{self.save_path}/bedgraph', exist_ok = True)

    def plot(self, vmin = -1, vmax = 1):
        import matplotlib.pyplot as plt
        height = 3
        width = 1 * np.log2(len(self.impact_score))
        fig, ax = plt.subplots(figsize = (width, height))
        self.plot_track(ax, self.impact_score, self.screen_start, self.step_size)
        self.reformat_ticks(plt)
        return plt

    def reformat_ticks(self, plt):
        # Format labels
        plt.xlabel('Genomic position (Mb)')

    def save_data(self, plt, save_pred, save_deletion, save_diff, save_impact_score, save_bedgraph):
        if self.plot_impact_score:
            plt.savefig(f'{self.save_path}/imgs/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}.png', bbox_inches = 'tight')
            plt.close()
        if save_pred:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_pred', self.tensor_pred)
        if save_deletion:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_perturbed', self.tensor_deletion)
        if save_diff:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_diff', self.tensor_diff)
        if save_impact_score:
            np.save(f'{self.save_path}/npy/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_impact_score', self.impact_score)
        if save_bedgraph:
            bedgraph_path = f'{self.save_path}/bedgraph/{self.chr_name}_screen_{self.screen_start}_{self.screen_end}_width_{self.perturb_width}_step_{self.step_size}_impact_score.bedgraph'
            self.save_bedgraph(self.chr_name, self.perturb_starts, self.perturb_ends, self.impact_score, bedgraph_path)

    def plot_track(self, ax, data, start, step):
        x = (np.array(range(len(data))) + int(start / step)) * step / 1000000
        width = min(self.perturb_width, int(step * 0.9)) / 1000000
        ax.bar(x, data, width = width)
        ax.margins(x=0)
        #ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('Impact score')
        #ax.set_ylim(-1, 8)

    def save_bedgraph(self, chr_name, starts, ends, scores, output_file):
        df = pd.DataFrame({'chr': chr_name, 'start': starts, 'end': ends, 'score': scores})
        df.to_csv(output_file, sep = '\t', index = False, header = False)


class MatrixPlotDeletionComparison:
    """
    三面板布局：删除前、删除后、差异，每个面板下方有对应的 insulation score track
    删除区域用红色标记
    """
    
    def __init__(self, output_path, pred_original, pred_deletion, pred_diff,
                 insulation_original, insulation_deletion, insulation_diff,
                 prefix, celltype, chr_name, start_pos,
                 deletion_start, deletion_width, deletion_bin_start, deletion_bin_end,
                 padding_type, show_deletion_line=True):
        self.output_path = output_path
        self.pred_original = pred_original
        self.pred_deletion = pred_deletion
        self.pred_diff = pred_diff
        self.insulation_original = insulation_original
        self.insulation_deletion = insulation_deletion
        self.insulation_diff = insulation_diff
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos
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
    
    def get_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])
        return color_map
    
    def get_diff_colormap(self):
        return 'RdBu_r'
    
    def rescale_coordinates(self, coords, zero_position):
        scaling_ratio = 8192
        replaced_coords = coords * scaling_ratio + zero_position
        coords_mb = replaced_coords / 1000000
        str_list = [f'{item:.1f}' for item in coords_mb]
        return str_list
    
    def plot(self, vmin=0, vmax=1, diff_vmin=-0.5, diff_vmax=0.5):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Rectangle
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # 创建 3x2 布局（3列 x 2行），上方 heatmap，下方 track
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1], hspace=0.05, wspace=0.3)
        
        color_map = self.get_colormap()
        diff_cmap = self.get_diff_colormap()
        
        # 数据和标题
        matrices = [self.pred_original, self.pred_deletion, self.pred_diff]
        insulations = [self.insulation_original, self.insulation_deletion, self.insulation_diff]
        titles = ['Before Deletion', 'After Deletion', 'Difference']
        cmaps = [color_map, color_map, diff_cmap]
        vmins = [vmin, vmin, diff_vmin]
        vmaxs = [vmax, vmax, diff_vmax]
        
        axes_heatmap = []
        axes_track = []
        
        for col in range(3):
            # 上方：Heatmap
            ax_heatmap = fig.add_subplot(gs[0, col])
            im = ax_heatmap.imshow(matrices[col], cmap=cmaps[col], vmin=vmins[col], vmax=vmaxs[col], aspect='equal')
            ax_heatmap.set_title(titles[col], fontsize=12, fontweight='bold')
            
            # 添加删除线标记（仅在前两张图）
            if self.show_deletion_line and col < 2:
                ax_heatmap.axvline(x=self.deletion_bin_start, color='black', linestyle='--', alpha=0.7, linewidth=1)
                ax_heatmap.axhline(y=self.deletion_bin_start, color='black', linestyle='--', alpha=0.7, linewidth=1)
                if self.deletion_bin_end < len(matrices[col]):
                    ax_heatmap.axvline(x=self.deletion_bin_end, color='black', linestyle='--', alpha=0.7, linewidth=1)
                    ax_heatmap.axhline(y=self.deletion_bin_end, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
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
            
            axes_heatmap.append(ax_heatmap)
            
            # 下方：Insulation Score Track（严格对齐 heatmap）
            ax_track = fig.add_subplot(gs[1, col])
            x = np.arange(len(insulations[col]))
            
            # 绘制 insulation score，删除区域用红色
            if col < 2:  # Original 和 Deletion
                # 非删除区域（蓝色）
                mask_deletion = (x >= self.deletion_bin_start) & (x < self.deletion_bin_end)
                
                # 先画整体填充（浅色）
                ax_track.fill_between(x, insulations[col], alpha=0.3, color='steelblue')
                
                # 删除区域用红色覆盖
                if np.any(mask_deletion):
                    ax_track.fill_between(x, insulations[col], where=mask_deletion, 
                                         alpha=0.7, color='crimson', label='Deleted region')
                
                # 绘制线条
                ax_track.plot(x, insulations[col], color='steelblue', linewidth=0.8)
                
                # 在删除区域画红色高亮背景
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
            
            axes_track.append(ax_track)
        
        # 添加整体标题
        deletion_mb_start = self.deletion_start / 1000000
        deletion_mb_end = (self.deletion_start + self.deletion_width) / 1000000
        fig.suptitle(f'{self.chr_name}: Deletion {deletion_mb_start:.3f} - {deletion_mb_end:.3f} Mb '
                    f'(width: {self.deletion_width:,} bp)', fontsize=14, fontweight='bold', y=1.02)
        
        self.save_data(plt, fig)
        return fig
    
    def save_data(self, plt, fig):
        # 保存图像
        filename_base = f'{self.chr_name}_{self.start_pos}_del_{self.deletion_start}_{self.deletion_width}_comparison'
        fig.savefig(f'{self.save_path}/imgs/{filename_base}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # 保存 numpy 数据
        np.save(f'{self.save_path}/npy/{filename_base}_original', self.pred_original)
        np.save(f'{self.save_path}/npy/{filename_base}_deletion', self.pred_deletion)
        np.save(f'{self.save_path}/npy/{filename_base}_diff', self.pred_diff)
        np.save(f'{self.save_path}/npy/{filename_base}_insulation_original', self.insulation_original)
        np.save(f'{self.save_path}/npy/{filename_base}_insulation_deletion', self.insulation_deletion)
        np.save(f'{self.save_path}/npy/{filename_base}_insulation_diff', self.insulation_diff)


class ComparisonPDFPlot:
    """
    3-panel comparison PDF: Condition A, Condition B, Difference.
    Heatmaps are rasterized at 500 DPI; text/axes remain vector in the PDF.
    """

    WINDOW = 2097152
    BIN_SIZE = 8192

    def __init__(self, output_path, pred_a, pred_b, pred_diff,
                 title_a, title_b, prefix, celltype, chr_name, start_pos,
                 filename_tag='comparison'):
        self.pred_a = pred_a
        self.pred_b = pred_b
        self.pred_diff = pred_diff
        self.title_a = title_a
        self.title_b = title_b
        self.prefix = prefix
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.filename_tag = filename_tag

        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok=True)
        os.makedirs(f'{self.save_path}/npy', exist_ok=True)

    def _rescale_coordinates(self, coords):
        coords_mb = (coords * self.BIN_SIZE + self.start_pos) / 1e6
        return [f'{v:.2f}' for v in coords_mb]

    def _get_hic_cmap(self):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])

    def plot(self, vmin=0, vmax=3, diff_vmin=-0.5, diff_vmax=0.5):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        hic_cmap = self._get_hic_cmap()
        diff_cmap = 'RdBu_r'

        panels = [
            (self.pred_a, self.title_a, hic_cmap, vmin, vmax),
            (self.pred_b, self.title_b, hic_cmap, vmin, vmax),
            (self.pred_diff, 'Difference', diff_cmap, diff_vmin, diff_vmax),
        ]

        fig = plt.figure(figsize=(16, 5.5))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)

        n_bins = self.pred_a.shape[0]
        tick_pos = np.arange(0, n_bins, 50)
        tick_labels = self._rescale_coordinates(tick_pos)

        for col, (matrix, title, cmap, vm, vx) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            im = ax.imshow(matrix, cmap=cmap, vmin=vm, vmax=vx,
                           aspect='equal', rasterized=True)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_xlim(-0.5, n_bins - 0.5)
            ax.set_ylim(n_bins - 0.5, -0.5)
            if col == 0:
                ax.set_ylabel('Genomic position (Mb)', fontsize=10)
            end_mb = (self.start_pos + self.WINDOW) / 1e6
            start_mb = self.start_pos / 1e6
            ax.set_xlabel(f'{self.chr_name}: {start_mb:.2f} - {end_mb:.2f} Mb', fontsize=9)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im, cax=cax).ax.tick_params(labelsize=7)

        fig.suptitle(f'{self.chr_name}  {self.start_pos:,} bp  —  {self.filename_tag}',
                     fontsize=13, fontweight='bold', y=1.02)

        self._save(fig, plt)
        return fig

    def _save(self, fig, plt):
        base = f'{self.chr_name}_{self.start_pos}_{self.filename_tag}'
        fig.savefig(f'{self.save_path}/imgs/{base}.pdf',
                    format='pdf', dpi=500, bbox_inches='tight')
        plt.close(fig)
        np.save(f'{self.save_path}/npy/{base}_a', self.pred_a)
        np.save(f'{self.save_path}/npy/{base}_b', self.pred_b)
        np.save(f'{self.save_path}/npy/{base}_diff', self.pred_diff)


class GlobalTrackPDFPlot:
    """
    Global impact-score track as a PDF bar chart + bedgraph output.
    """

    def __init__(self, output_path, chr_name, celltype, prefix,
                 window_starts, window_ends, impact_scores, step_size,
                 filename_tag='global_track'):
        self.chr_name = chr_name
        self.celltype = celltype
        self.prefix = prefix
        self.window_starts = np.asarray(window_starts)
        self.window_ends = np.asarray(window_ends)
        self.impact_scores = np.asarray(impact_scores)
        self.step_size = step_size
        self.filename_tag = filename_tag

        self.save_path = f'{output_path}/{celltype}/{prefix}'
        os.makedirs(f'{self.save_path}/imgs', exist_ok=True)
        os.makedirs(f'{self.save_path}/npy', exist_ok=True)
        os.makedirs(f'{self.save_path}/bedgraph', exist_ok=True)

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = len(self.impact_scores)
        width_inch = max(6, np.log2(max(n, 2)) * 1.5)
        fig, ax = plt.subplots(figsize=(width_inch, 3))

        x_mb = self.window_starts / 1e6
        bar_w = self.step_size * 0.9 / 1e6
        ax.bar(x_mb, self.impact_scores, width=bar_w, color='steelblue', edgecolor='none')
        ax.margins(x=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Impact score')
        ax.set_xlabel(f'{self.chr_name}  Genomic position (Mb)')
        ax.set_title(f'{self.filename_tag}  (step={self.step_size:,} bp)',
                     fontsize=11, fontweight='bold')

        base = f'{self.chr_name}_{self.filename_tag}'
        fig.savefig(f'{self.save_path}/imgs/{base}.pdf',
                    format='pdf', bbox_inches='tight')
        plt.close(fig)

        np.save(f'{self.save_path}/npy/{base}_impact_scores', self.impact_scores)
        np.save(f'{self.save_path}/npy/{base}_window_starts', self.window_starts)

        bg_path = f'{self.save_path}/bedgraph/{base}.bedgraph'
        df = pd.DataFrame({
            'chr': self.chr_name,
            'start': self.window_starts,
            'end': self.window_ends,
            'score': self.impact_scores,
        })
        df.to_csv(bg_path, sep='\t', index=False, header=False)
        print(f"Global track saved to {self.save_path}")


class MultiTFComparisonPlot:
    """
    N-panel comparison: one heatmap per TF prediction at the same locus.
    Saves PDF (rasterized heatmaps, 500 DPI) + per-TF .npy files.
    """

    WINDOW = 2097152
    BIN_SIZE = 8192

    def __init__(self, output_path, predictions, tf_names,
                 celltype, chr_name, start_pos, filename_tag='multi_tf'):
        self.predictions = predictions  # list of (256,256) arrays
        self.tf_names = tf_names
        self.celltype = celltype
        self.chr_name = chr_name
        self.start_pos = start_pos
        self.filename_tag = filename_tag

        self.save_path = f'{output_path}/{celltype}/multi_tf'
        os.makedirs(f'{self.save_path}/imgs', exist_ok=True)
        os.makedirs(f'{self.save_path}/npy', exist_ok=True)

    def _rescale_coordinates(self, coords):
        coords_mb = (coords * self.BIN_SIZE + self.start_pos) / 1e6
        return [f'{v:.2f}' for v in coords_mb]

    def _get_hic_cmap(self):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("bright_red", [(1, 1, 1), (1, 0, 0)])

    def plot(self, vmin=0, vmax=3):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        n = len(self.predictions)
        hic_cmap = self._get_hic_cmap()
        fig = plt.figure(figsize=(4.5 * n, 5))
        gs = gridspec.GridSpec(1, n, wspace=0.3)

        n_bins = self.predictions[0].shape[0]
        tick_pos = np.arange(0, n_bins, 50)
        tick_labels = self._rescale_coordinates(tick_pos)
        start_mb = self.start_pos / 1e6
        end_mb = (self.start_pos + self.WINDOW) / 1e6

        for col, (matrix, tf_name) in enumerate(zip(self.predictions, self.tf_names)):
            ax = fig.add_subplot(gs[0, col])
            im = ax.imshow(matrix, cmap=hic_cmap, vmin=vmin, vmax=vmax,
                           aspect='equal', rasterized=True)
            ax.set_title(tf_name, fontsize=12, fontweight='bold')
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha='right')
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=7)
            ax.set_xlim(-0.5, n_bins - 0.5)
            ax.set_ylim(n_bins - 0.5, -0.5)
            if col == 0:
                ax.set_ylabel('Genomic position (Mb)', fontsize=9)
            ax.set_xlabel(f'{self.chr_name}: {start_mb:.1f}-{end_mb:.1f} Mb', fontsize=8)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            plt.colorbar(im, cax=cax).ax.tick_params(labelsize=6)

        fig.suptitle(f'{self.chr_name}  {self.start_pos:,} bp  —  {self.filename_tag}',
                     fontsize=13, fontweight='bold', y=1.02)

        base = f'{self.chr_name}_{self.start_pos}_{self.filename_tag}'
        fig.savefig(f'{self.save_path}/imgs/{base}.pdf',
                    format='pdf', dpi=500, bbox_inches='tight')
        plt.close(fig)
        for matrix, tf_name in zip(self.predictions, self.tf_names):
            np.save(f'{self.save_path}/npy/{base}_{tf_name}', matrix)
        return fig


def plot_cross_tf_matrix(matrix, tf_names, chr_name, output_path, filename_tag='cross_tf_matrix'):
    """Plot a symmetric cross-TF impact score matrix as a heatmap PDF."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')
    n = len(tf_names)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tf_names, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(tf_names, fontsize=10)
    for i in range(n):
        for j in range(n):
            color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.4f}', ha='center', va='center',
                    fontsize=9, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean impact score')
    ax.set_title(f'Cross-TF Impact Matrix — {chr_name}', fontsize=13, fontweight='bold')
    fig.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    fig.savefig(f'{output_path}/{chr_name}_{filename_tag}.pdf',
                format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Matrix heatmap saved to {output_path}/{chr_name}_{filename_tag}.pdf")

