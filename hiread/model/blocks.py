import torch
import torch.nn as nn
import numpy as np
import copy
import einops
import math
from timm.models.layers import trunc_normal_
from .RRT.modules.rrt import RRTEncoder
import functools
import torch.nn.functional as F
from scipy import ndimage


class PPEG(nn.Module):
    def __init__(self, dim=256,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class PEG(nn.Module):
    def __init__(self, dim=256,k=7,bias=True,conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length >0:
            x = x[:,:-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


# class SINCOS(nn.Module):
#     def __init__(self,embed_dim=512):
#         super(SINCOS, self).__init__()
#         self.embed_dim = embed_dim
#         self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)
#     def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
#         """
#         embed_dim: output dimension for each position
#         pos: a list of positions to be encoded: size (M,)
#         out: (M, D)
#         """
#         assert embed_dim % 2 == 0
#         omega = np.arange(embed_dim // 2, dtype=np.float)
#         omega /= embed_dim / 2.
#         omega = 1. / 10000**omega  # (D/2,)

#         pos = pos.reshape(-1)  # (M,)
#         out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

#         emb_sin = np.sin(out) # (M, D/2)
#         emb_cos = np.cos(out) # (M, D/2)

#         emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#         return emb

#     def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid):
#         assert embed_dim % 2 == 0

#         # use half of dimensions to encode grid_h
#         emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#         emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#         emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#         return emb

#     def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
#         """
#         grid_size: int of the grid height and width
#         return:
#         pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#         """
#         grid_h = np.arange(grid_size, dtype=np.float32)
#         grid_w = np.arange(grid_size, dtype=np.float32)
#         grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#         grid = np.stack(grid, axis=0)

#         grid = grid.reshape([2, 1, grid_size, grid_size])
#         pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#         if cls_token:
#             pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
#         return pos_embed

#     def forward(self, x):
#         B, N, C = x.shape
#         # B,H,W,C = x.shape
#         # padding
#         H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
#         add_length = H * W - N
#         x = torch.cat([x, x[:,:add_length,:]],dim = 1)

#         pos_embed = torch.zeros(1, H * W + 1, self.embed_dim)
#         pos_embed = self.get_2d_sincos_pos_embed(pos_embed.shape[-1], int(H), cls_token=True)
#         pos_embed = torch.from_numpy(self.pos_embed).float().unsqueeze(0).to(x.device)

#         # pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)

#         # print(pos_embed.size())
#         # print(x.size())
#         # x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        

#         x = x + pos_embed[:, 1:, :]

#         if add_length >0:
#             x = x[:,:-add_length]

#         return x

# class APE(nn.Module):
#     def __init__(self,embed_dim=512,num_patches=64):
#         super(APE, self).__init__()
#         self.absolute_pos_embed = nn.Parameter(torch.zeros( num_patches, embed_dim))
#         trunc_normal_(self.absolute_pos_embed, std=.02)
    
#     def forward(self, x):
#         B,H,W,C = x.shape
#         return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)

# class RPE(nn.Module):
#     def __init__(self,num_heads=8,region_size=(8,8)):
#         super(RPE, self).__init__()
#         self.region_size = region_size

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * region_size[0] - 1) * (2 * region_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the region
#         coords_h = torch.arange(region_size[0])
#         coords_w = torch.arange(region_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += region_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += region_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * region_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)
#         trunc_normal_(self.relative_position_bias_table, std=.02)
    
#     def forward(self, x):
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         print(relative_position_bias.size())

#         return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)

class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, output_size = 256 , filter_size = 5, num_blocks = 4):#output_size = 256
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 32, 3, 2, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]#
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]#
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)#256

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class EncoderSplit(Encoder):
    def __init__(self, num_epi, output_size = 256, filter_size = 5, num_blocks = 12):#output_size = 256,num_blocks = 12
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(num_epi, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
 
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.conv_end = nn.Conv1d(256, output_size, 1)#256
        # 定义门控层
        # self.gate = nn.Linear(256*2, 2,bias = True)  # 假设合并后的特征维度为256 * 2
        # self.gate_fusion = gatedFusion(256)
    def forward(self, x):

        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))

        x = torch.cat([seq, epi], dim = 1)
        # print(x.size())
        # x = self.gate_fusion(seq, epi)
        
        out = self.conv_end(x)
        return out
    
# class LoopAwareConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(LoopAwareConvBlock, self).__init__()
#         # 标准卷积路径
#         self.standard_path = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )
        
#         # 扩张卷积路径 - 捕捉更广范围的依赖关系
#         self.dilated_path = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )
        
#         # 长距离卷积路径 - 用更大的卷积核
#         self.long_range_path = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )
        
#         # 特征融合
#         self.fusion = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)
        
#         # 若输入输出通道不同，增加残差连接的映射
#         self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
#     def forward(self, x):
#         # 三种不同感受野的卷积路径
#         standard_out = self.standard_path(x)
#         dilated_out = self.dilated_path(x)
#         long_range_out = self.long_range_path(x)
        
#         # 特征融合
#         combined = torch.cat([standard_out, dilated_out, long_range_out], dim=1)
#         fused = self.fusion(combined)
        
#         # 残差连接
#         residual = self.residual(x)
#         return fused + residual

# class EnhancedGatedFusion(nn.Module):
#     def __init__(self, dim):
#         super(EnhancedGatedFusion, self).__init__()
        
#         # 门控计算
#         self.fc1 = nn.Linear(dim, dim, bias=True)
#         self.fc2 = nn.Linear(dim, dim, bias=True)
        
#         # 特征增强路径
#         self.enhance1 = nn.Sequential(
#             nn.Conv1d(dim, dim, kernel_size=1),
#             nn.ReLU()
#         )
#         self.enhance2 = nn.Sequential(
#             nn.Conv1d(dim, dim, kernel_size=1),
#             nn.ReLU()
#         )
        
#         # 跨特征注意力
#         self.cross_attn1 = nn.Conv1d(dim, dim, kernel_size=1)
#         self.cross_attn2 = nn.Conv1d(dim, dim, kernel_size=1)
        
#     def forward(self, x1, x2):
#         # 原始特征维度调整
#         batch_size, channels, seq_len = x1.size()
#         x1_flat = x1.transpose(1, 2).reshape(-1, channels)
#         x2_flat = x2.transpose(1, 2).reshape(-1, channels)
        
#         # 计算门控权重
#         x11 = self.fc1(x1_flat)
#         x22 = self.fc2(x2_flat)
        
#         # 生成门控信号
#         z = torch.sigmoid(x11 + x22)
        
#         # 恢复原始形状
#         z = z.view(batch_size, seq_len, channels).transpose(1, 2)
        
#         # 特征增强
#         x1_enhanced = self.enhance1(x1)
#         x2_enhanced = self.enhance2(x2)
        
#         # 跨特征注意力 - x1影响x2，x2影响x1
#         x1_attention = torch.sigmoid(self.cross_attn1(x2))
#         x2_attention = torch.sigmoid(self.cross_attn2(x1))
        
#         x1_final = x1_enhanced * (1 + x1_attention)  # 增强x1中与x2相关的部分
#         x2_final = x2_enhanced * (1 + x2_attention)  # 增强x2中与x1相关的部分
        
#         # 门控混合
#         gated_x1 = z * x1_final
#         gated_x2 = (1-z) * x2_final
        
#         # 拼接输出，保持形状一致
#         out = torch.cat([gated_x1, gated_x2], dim=1)
#         return out
    
    
class gatedFusion(nn.Module):

    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        x11 = self.fc1(x1)
        x22 = self.fc2(x2)
        # 通过门控单元生成权重表示
        z = torch.sigmoid(x11+x22)
        # 对两部分输入执行加权和
        # out = z*x1 + (1-z)*x2
        out = torch.cat([z*x1,(1-z)*x2], dim = 1)
        return out
    
# class MultiScaleConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], stride=2):
#         super(MultiScaleConv1d, self).__init__()
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=stride, padding=k//2),
#                 nn.BatchNorm1d(out_channels),
#                 nn.ReLU()
#             ) for k in kernel_sizes
#         ])
#         # 融合各分支特征，采用1×1卷积进行通道数调整
#         self.fusion = nn.Conv1d(len(kernel_sizes)*out_channels, out_channels, kernel_size=1)
    
#     def forward(self, x):
#         branch_outputs = [branch(x) for branch in self.branches]
#         # 拼接通道维度：[B, out_channels * num_branches, L]
#         x_cat = torch.cat(branch_outputs, dim=1)
#         # 融合
#         x_fused = self.fusion(x_cat)
#         return x_fused
# class ImprovedMultiScaleConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 11], stride=1):
#         super(ImprovedMultiScaleConv1d, self).__init__()
#         self.branches = nn.ModuleList()
#         self.branch_count = len(kernel_sizes)
        
#         # 确保每个分支输出的通道数加起来不超过 out_channels
#         channels_per_branch = out_channels // self.branch_count
        
#         # 添加多种尺度的卷积分支
#         for k in kernel_sizes:
#             self.branches.append(nn.Sequential(
#                 nn.Conv1d(in_channels, channels_per_branch, kernel_size=k, stride=stride, 
#                          padding=k//2),
#                 nn.BatchNorm1d(channels_per_branch),
#                 nn.ReLU()
#             ))
        
#         # 简化注意力机制，避免维度扩展
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),  # 全局平均池化减少序列长度维度
#             nn.Conv1d(self.branch_count * channels_per_branch, self.branch_count, kernel_size=1),
#             nn.Softmax(dim=1)
#         )
        
#         # 确保最终输出通道数严格等于 out_channels
#         self.fusion = nn.Sequential(
#             nn.Conv1d(self.branch_count * channels_per_branch, out_channels, kernel_size=1),
#             nn.BatchNorm1d(out_channels)
#         )
    
#     def forward(self, x):
#         # 获取各分支输出
#         branch_outputs = [branch(x) for branch in self.branches]
        
#         # 拼接所有分支输出
#         x_cat = torch.cat(branch_outputs, dim=1)
        
#         # 直接进行特征融合，确保维度正确
#         x_fused = self.fusion(x_cat)
        
#         return x_fused


# class RRTEncoderSplit(nn.Module):
#     def __init__(self, num_epi, output_size=256, filter_size=5, num_blocks=12,
#                 mlp_dim=256, attn='crmsa', region_num=16, n_layers=4, n_heads=8, pos='PEG'):
#         super(RRTEncoderSplit, self).__init__()
#         self.filter_size = filter_size
        
#         # 序列特征处理初始层
#         self.conv_start_seq = nn.Sequential(
#             nn.Conv1d(5, 16, 3, 2, 1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#         )
        
#         # 表观遗传特征处理初始层
#         self.conv_start_epi = nn.Sequential(
#             nn.Conv1d(num_epi, 16, 3, 2, 1),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#         )
        
#         # 隐层大小设置
#         hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
#         hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        
#         hiddens_half = (np.array(hiddens) / 2).astype(int)
#         hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        
#         # 构建残差块 - 修改这部分
#         # self.res_blocks_seq = self.get_improved_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
#         # self.res_blocks_epi = self.get_improved_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
#         self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
#         self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)       
#         # 新增: Loop感知模块 - 在高层特征上应用
#         self.loop_aware_seq = LoopAwareConvBlock(128, 128)  # 适用于中间层特征
#         self.loop_aware_epi = LoopAwareConvBlock(128, 128)
        
#         # RRTEncoder部分保持不变
#         self.encoder_seq = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, 
#                                     n_layers=n_layers, n_heads=n_heads)
#         self.encoder_epi = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, 
#                                      n_layers=n_layers, n_heads=n_heads)
        
#         # 修改: 使用改进的门控融合替换原有的gatedFusion
#         self.gate_fusion = gatedFusion(256)  # 替换原有的gatedFusion
        
#         # 输出层
#         self.conv_end = nn.Conv1d(256, output_size, 1)  # 注意这里输入通道是512，因为门控融合会拼接特征
    
#     # 新方法：构建改进的残差块
#     def get_improved_res_blocks(self, n, his, hs):
#         blocks = []
        
#         for i, h, hi in zip(range(n), hs, his):
#             # 前几层使用普通卷积块
#             if i < n // 2:
#                 blocks.append(ConvBlock(self.filter_size, hidden_in=hi, hidden=h))
#             # 后几层使用改进的多尺度卷积，以捕捉更复杂的特征
#             else:
#                 blocks.append(nn.Sequential(
#                     ImprovedMultiScaleConv1d(hi, h, stride=1),
#                     nn.Dropout(0.1)
#                 ))
        
#         return nn.Sequential(*blocks)
    
#     def forward(self, x):
#         # 分离序列和表观遗传学特征
#         seq = x[:, :5, :]
#         epi = x[:, 5:, :]
        
#         # 初始卷积处理
#         seq = self.conv_start_seq(seq)
#         print('seq_size'+str(seq.size()))
#         epi = self.conv_start_epi(epi)
#         print('epi_size'+str(epi.size()))
#         # 残差块处理
#         seq = self.res_blocks_seq(seq)
#         print('seq_size'+str(seq.size()))
#         epi = self.res_blocks_epi(epi)
#         print('epi_size'+str(epi.size()))
#         # 新增: 应用Loop感知模块
#         # 提取中间层特征进行处理
#         if seq.size(1) >= 128:
#             # 提取128通道特征
#             seq_loop = seq[:, :128, :]
#             epi_loop = epi[:, :128, :]
            
#             # 应用Loop感知处理
#             seq_loop = self.loop_aware_seq(seq_loop)
#             epi_loop = self.loop_aware_epi(epi_loop)
            
#             # 替换回原始特征的对应部分
#             seq[:, :128, :] = seq_loop
#             epi[:, :128, :] = epi_loop
        
#         # RRTEncoder处理
#         seq = self.encoder_seq(seq)
#         print('seq_size'+str(seq.size()))
#         epi = self.encoder_epi(epi)
#         print('epi_size'+str(epi.size()))
#         # 使用改进的门控融合
#         x = self.gate_fusion(seq, epi)
        
#         # 输出层处理
#         out = self.conv_end(x)
        
#         return out
#     def get_res_blocks(self, n, his, hs):
#         blocks = []
#         for i, h, hi in zip(range(n), hs, his):
#             blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
#         res_blocks = nn.Sequential(*blocks)
#         return res_blocks    
    
    
class RRTEncoderSplit(nn.Module):
    def __init__(self, num_epi, output_size = 256, filter_size = 5, num_blocks = 12,mlp_dim=256, attn='crmsa', region_num=16, n_layers=4, n_heads=8,pos='PEG', epeg=True):#output_size = 256,num_blocks = 12
        super(RRTEncoderSplit, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(num_epi, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
 
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.encoder_seq = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, n_layers=n_layers, n_heads=n_heads, pos=pos, epeg=epeg)
        self.encoder_epi = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, n_layers=n_layers, n_heads=n_heads, pos=pos, epeg=epeg)
        self.conv_end = nn.Conv1d(256, output_size, 1)#256
        # 定义门控层
        # self.gate = nn.Linear(256*2, 2,bias = True)  # 假设合并后的特征维度为256 * 2
        self.gate_fusion = gatedFusion(256)
        #定义norm
        # self.crossnorm = CrossNorm()
        # self.selfnorm = SelfNorm(chan_num=1)
        # self.normblock = CNSN(self.crossnorm, self.selfnorm)
    def forward(self, x, epi=None, need_attn=False):
        # Support both ``forward(x)`` and ``forward(seq, epi)`` calls.
        if epi is None:
            seq = x[:, :5, :]
            epi = x[:, 5:, :]
        else:
            seq = x
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        
        # 通过各自的RRTEncoder处理每部分数据
        if need_attn:
            seq, seq_attn = self.encoder_seq(seq, need_attn=True)
            epi, epi_attn = self.encoder_epi(epi, need_attn=True)
            # 合并attention权重
            attn_weights = {
                'seq_attention': seq_attn,
                'epi_attention': epi_attn
            }
        else:
            seq = self.encoder_seq(seq)
            epi = self.encoder_epi(epi)
            attn_weights = None
        
        x = self.gate_fusion(seq, epi)
        # x = x.unsqueeze(1)
        # x = self.normblock(x)
        # x = x.squeeze(1)
        out = self.conv_end(x)
        
        if need_attn:
            return out, attn_weights
        else:
            return out
    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    

# class RRTEncoderSplitMultiScale(nn.Module):
#     def __init__(self, num_epi, output_size=256, filter_size=5, num_blocks=12, mlp_dim=256,
#                  attn='crmsa', region_num=16, n_layers=4, n_heads=8, pos='PEG'):
#         super(RRTEncoderSplitMultiScale, self).__init__()
#         # 使用多尺度卷积模块，输入通道为原始数据通道数（例如序列部分为5）
#         self.multiscale_conv_seq = MultiScaleConv1d(in_channels=5, out_channels=16, kernel_sizes=[3, 5, 7], stride=2)
#         self.multiscale_conv_epi = MultiScaleConv1d(in_channels=num_epi, out_channels=16, kernel_sizes=[3, 5, 7], stride=2)
        
#         # 下面可以继续使用残差模块，也可以在残差模块中引入空洞卷积来进一步扩展感受野
#         hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
#         hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
#         hiddens_half = (np.array(hiddens) / 2).astype(int)
#         hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
#         self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
#         self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        
#         self.encoder_seq = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, n_layers=n_layers, n_heads=n_heads)
#         self.encoder_epi = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, n_layers=n_layers, n_heads=n_heads)
#         self.conv_end = nn.Conv1d(256, output_size, 1)
#         self.gate_fusion = gatedFusion(256)
    
#     def forward(self, x):
#         seq = x[:, :5, :]
#         epi = x[:, 5:, :]
#         seq = self.res_blocks_seq(self.multiscale_conv_seq(seq))
#         epi = self.res_blocks_epi(self.multiscale_conv_epi(epi))
        
#         # 分别用 RRTEncoder 处理两部分特征
#         seq_encoded = self.encoder_seq(seq)
#         epi_encoded = self.encoder_epi(epi)
        
#         # 此处可尝试多种融合策略：简单拼接、门控融合等
#         x_fused = self.gate_fusion(seq, epi)
#         out = self.conv_end(x_fused)
#         return out
    
#     def get_res_blocks(self, n, his, hs):
#         blocks = []
#         for i, h, hi in zip(range(n), hs, his):
#             blocks.append(ConvBlock(5, hidden_in=hi, hidden=h))
#         return nn.Sequential(*blocks)


    
    
    
# ===============================
# 原有的ConvBlock、Encoder、Decoder等代码
# （这里省略，保持不变）
# ===============================

# --- 新增：条件扩散的UNet模块 ---
# class ConditionalDiffusionUNet(nn.Module):
#     """
#     一个简单的UNet结构，用于条件扩散生成（根据embedding条件生成Hi-C图像）。
#     输入：noisy图像（例如：B x 1 x H x W）和条件向量（B x cond_dim）
#     输出：预测的噪声（或预测去噪后的图像，可按训练目标调整）
#     """
#     def __init__(self, in_channels=1, cond_dim=256, base_channels=64):
#         super(ConditionalDiffusionUNet, self).__init__()
#         # 编码器部分
#         self.enc_conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(),
#         )
#         self.enc_conv2 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.ReLU(),
#         )
#         # 将条件向量映射到与中间特征同维度（用于条件注入，例如FiLM方式）
#         self.cond_proj = nn.Linear(cond_dim, base_channels * 2 * 2)  # 生成scale和bias

#         # 解码器部分
#         self.dec_conv1 = nn.Sequential(
#             nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(base_channels),
#             nn.ReLU(),
#         )
#         self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

#     def forward(self, x, cond):
#         """
#         x: noisy image, shape (B, 1, H, W)
#         cond: embedding, shape (B, cond_dim)
#         """
#         # 编码
#         enc1 = self.enc_conv1(x)          # (B, base_channels, H, W)
#         enc2 = self.enc_conv2(enc1)        # (B, base_channels*2, H/2, W/2)

#         # 条件注入：将cond通过全连接映射后分为scale和bias
#         cond_params = self.cond_proj(cond)  # (B, base_channels*2*2)
#         cond_params = cond_params.view(-1, 2, enc2.size(1), 1, 1)  # (B, 2, base_channels*2, 1, 1)
#         scale, bias = cond_params[:,0], cond_params[:,1]
#         # 调整enc2特征（简单的FiLM机制）
#         cond_enc = enc2 * scale + bias

#         # 解码
#         dec1 = self.dec_conv1(cond_enc)  # (B, base_channels, H, W)
#         # 可加入skip connection
#         dec1 = dec1 + enc1
#         out = self.out_conv(dec1)  # 输出图像，形状(B, 1, H, W)
#         return out

# # --- 新增：扩散调度器，用于添加噪声 ---
# class DiffusionScheduler:
#     """
#     简单的扩散调度器
#     """
#     def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
#         self.num_timesteps = num_timesteps
#         self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
#         self.alpha = 1 - self.beta
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)

#     def add_noise(self, x0, t, noise=None):
#         """
#         x0: 原图像，形状 (B, C, H, W)
#         t: 时间步（tensor，shape: (B,)），每个样本的时间步
#         noise: 可选，若未提供则采样正态噪声
#         """
#         if noise is None:
#             noise = torch.randn_like(x0)
#         # 取出每个样本对应的alpha_bar
#         device = x0.device
#         alpha_bar_t = self.alpha_bar[t].to(device)  # (B,)
#         # 将alpha_bar_t扩展到图像尺寸
#         alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
#         noisy = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
#         return noisy

    
    
    
class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):#hidden = 256 
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

# class TransformerLayer(torch.nn.TransformerEncoderLayer):
#     # Pre-LN structure
    
#     def forward(self, src, src_mask = None, src_key_padding_mask = None):
#         # MHA section
#         src_norm = self.norm1(src)
#         src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
#                                     attn_mask=src_mask,
#                                     key_padding_mask=src_key_padding_mask)
#         src = src + self.dropout1(src_side)

#         # MLP section
#         src_norm = self.norm2(src)
#         src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
#         src = src + self.dropout2(src_side)
#         return src, attn_weights

# class TransformerEncoder(torch.nn.TransformerEncoder):

    # def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
#         super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
#         self.layers = self._get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.record_attn = record_attn
#     # def forward(self, src, mask = None, src_key_padding_mask = None):
#     def forward(self, src):
#         r"""Pass the input through the encoder layers in turn.

#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         output = src

#         attn_weight_list = []

#         for mod in self.layers:
#             # output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             # output, attn_weights = mod(output)
#             output = mod(output)
#             # attn_weight_list.append(attn_weights.unsqueeze(0).detach())
#         if self.norm is not None:
#             output = self.norm(output)

#         if self.record_attn:
#             return output, torch.cat(attn_weight_list)
#         else:
#             return output

#     def _get_clones(self, module, N):
#         return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])

# class PositionalEncoding(nn.Module):

#     def __init__(self, hidden, dropout = 0.1, max_len = 256):#max_len = 256
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
#         pe = torch.zeros(max_len, 1, hidden)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)





# class AttnModule(nn.Module):
#     def __init__(self, hidden = 128, layers = 1, record_attn = False, inpu_dim = 256):
#         super(AttnModule, self).__init__()

#         self.record_attn = record_attn
#         # self.pos_encoder = PositionalEncoding(hidden, dropout = 0.1)
#         # self.pos_encoder = PEG()
#         # encoder_layers = TransformerLayer(hidden, 
#         #                                   nhead = 8,
#         #                                   dropout = 0.1,
#         #                                   dim_feedforward = 512,
#         #                                   batch_first = True)
#         # encoder_layers = EfficientAdditiveAttnetion(in_dims=hidden,token_dim=256,num_heads=1,dropout= 0.5) 
#         encoder_layers = CrossRegionAttntion(dim=256, num_heads=8, region_size=16) 
#         self.module = TransformerEncoder(encoder_layers, 
#                                          layers, 
#                                          record_attn = record_attn)

#     def forward(self, x):
#         # x = self.pos_encoder(x)
#         # print(x.size())
#         output = self.module(x)
#         return output

#     def inference(self, x):
#         return self.module(x)

# class RRTEncoderSingle(nn.Module):
#     def __init__(self, in_channel, output_size = 256, filter_size = 7, num_blocks = 12,mlp_dim=256, attn='crmsa', region_num=32, n_layers=4, n_heads=8,pos='PEG'):#output_size = 256,num_blocks = 12
#         super(RRTEncoderSingle, self).__init__()
#         self.filter_size = filter_size
#         self.conv_start = nn.Sequential(
#                                     nn.Conv1d(in_channel, 32, 3, 2, 1),
#                                     nn.BatchNorm1d(32),
#                                     nn.ReLU(),
#                                     )
#         hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
#         hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
#         self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
#         self.encoder = RRTEncoder(mlp_dim=mlp_dim, attn=attn, region_num=region_num, n_layers=n_layers, n_heads=n_heads)
#         self.conv_end = nn.Conv1d(256, output_size, 1)#256
#     def forward(self, x):
#         x = self.conv_start(x)
#         x = self.res_blocks(x)
#         x = self.encoder(x)
#         out = self.conv_end(x)
#         return out
#     def get_res_blocks(self, n, his, hs):
#         blocks = []
#         for i, h, hi in zip(range(n), hs, his):
#             blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
#         res_blocks = nn.Sequential(*blocks)
#         return res_blocks

if __name__ == '__main__':
    main()
