import torch
import torch.nn as nn

from . import blocks

# 定义ConvModel类
class ConvModel(nn.Module):
    #mid_hidden = 256
    def __init__(self, num_genomic_features, mid_hidden = 256):
        '''
        初始化ConvModel
        '''
        super(ConvModel, self).__init__()
        # 初始化编码器
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = mid_hidden, num_blocks = 12)#num_blocks = 12
        # 初始化解码器
        self.decoder = blocks.Decoder(mid_hidden * 2)

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to: 
        bs, feat, img_len
        '''
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)#256
        # print(x_i.size())
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)#256
        # print(x_j.size())
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map

# 定义ConvTransModel类
class ConvTransModel(ConvModel):
    #mid_hidden = 256
    def __init__(self, num_genomic_features, mid_hidden = 256, record_attn = False, pos_embedding="EPEG"):
        '''
        初始化ConvTransModel
        '''
        super(ConvTransModel, self).__init__(num_genomic_features)
        pos_embedding_normalized = str(pos_embedding).upper()
        if pos_embedding_normalized == "EPEG":
            pos = "none"
            epeg = True
        elif pos_embedding_normalized == "PEG":
            pos = "PEG"
            epeg = True
        elif pos_embedding_normalized == "PPEG":
            pos = "PPEG"
            epeg = True
        elif pos_embedding_normalized == "NONE":
            pos = "none"
            epeg = False
        else:
            raise ValueError(f"Unsupported pos_embedding: {pos_embedding}")
        # 初始化编码器
        # ``EPEG`` here means the released checkpoint-compatible default:
        # attention blocks keep EPEG enabled while explicit PEG/PPEG positional
        # convolutions remain disabled.
        self.encoder = blocks.RRTEncoderSplit(num_genomic_features, output_size = mid_hidden,filter_size=7, num_blocks = 12,mlp_dim=256, attn='crmsa', region_num=32, n_layers=4, n_heads=8,pos=pos, epeg=epeg) #num_blocks =12
        # self.encoder = blocks.Encoder(num_genomic_features, output_size = mid_hidden, num_blocks = 32) #num_blocks =12
        # 初始化Attn模块
        # self.attn = blocks.AttnModule(hidden = mid_hidden, record_attn = record_attn)
        # 初始化解码器
        self.decoder = blocks.Decoder(mid_hidden * 2,num_blocks=12)
        # 设置record_attn
        self.record_attn = record_attn
    
    def forward(self, x, need_attn=None):
  
        x = self.move_feature_forward(x).float()
        x = x.float()

        # 确定是否需要attention权重（优先使用传入的参数）
        should_record_attn = need_attn if need_attn is not None else self.record_attn

        # 从encoder获取特征和attention权重
        if should_record_attn:
            x, attn_weights = self.encoder(x, need_attn=True)
        else:
            x = self.encoder(x)
            attn_weights = None

        # 对角化
        x = self.diagonalize(x)
        # 解码
        x = self.decoder(x).squeeze(1)
        
        # 返回结果
        if should_record_attn:
            return x, attn_weights
        else:
            return x
# class ConvTransModel(ConvModel):
#     def __init__(self, num_genomic_features, mid_hidden=256, record_attn=False, enable_loop_enhancement=True):
#         '''
#         初始化ConvTransModel，增加loop结构检测与增强功能
#         '''
#         super(ConvTransModel, self).__init__(num_genomic_features)
#         print('Initializing Enhanced ConvTransModel with Loop Detection')
        
#         # 初始化编码器 - 使用已修改的RRTEncoderSplit
#         self.encoder = blocks.RRTEncoderSplit(
#             num_genomic_features, 
#             output_size=mid_hidden, 
#             filter_size=7, 
#             num_blocks=12,
#             mlp_dim=256, 
#             attn='crmsa', 
#             region_num=32, 
#             n_layers=4, 
#             n_heads=8,
#             pos='PPEG'
#         )
        
#         # 新增：Loop特征增强模块
#         self.enable_loop_enhancement = enable_loop_enhancement
#         if enable_loop_enhancement:
#             # Loop特性增强层 - 在对角化前应用
#             self.loop_feature_enhancer = nn.Sequential(
#                 blocks.ImprovedMultiScaleConv1d(mid_hidden, mid_hidden, kernel_sizes=[3, 5, 7, 11]),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             )
            
#             # 对角线周边特征增强层 - 专注于loop所在的对角线周围区域
#             self.diagonal_enhancer = blocks.LoopAwareConvBlock(mid_hidden * 2, mid_hidden * 2)
        
#         # 解码器
#         self.decoder = blocks.Decoder(mid_hidden * 2, num_blocks=12)
        
#         # 设置record_attn
#         self.record_attn = record_attn
    
#     def forward(self, x):
#         # 移动特征前置处理
#         x = self.move_feature_forward(x).float()
        
#         # 通过编码器
#         x = self.encoder(x)
        
#         # 新增：应用Loop特征增强（如果启用）
#         if self.enable_loop_enhancement:
#             # 增强编码器输出的特征
#             x = self.loop_feature_enhancer(x)
        
#         # 对角化操作
#         x = self.diagonalize(x)
        
#         # 新增：对角线区域增强（如果启用）
#         if self.enable_loop_enhancement:
#             # 增强对角化后的特征，特别关注可能包含loop的区域
#             x = self.diagonal_enhancer(x)
        
#         # 解码器处理
#         x = self.decoder(x).squeeze(1)
        
#         # 返回结果
#         if self.record_attn:
#             # 注：这段代码实际上不会执行，因为我们已经移除了attn模块
#             # 保留此代码是为了与原始接口兼容
#             return x, None  # 替换了原来的attn_weights
#         else:
#             return x

if __name__ == '__main__':
    raise SystemExit("Import this module from training or inference code.")
