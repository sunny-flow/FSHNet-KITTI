import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ...utils.spconv_utils import replace_feature, spconv
from .scatterformer_utils import rearrange, scatter_matmul_kv, scatter_matmul_qc, torch_scatter
from torch.cuda.amp import autocast

    
#改进3：为SlotFormer增加可独立启停的ELGM层替换参数；默认值保持原两层注意力结构不变。
class SlotFormer(spconv.SparseModule):
    def __init__(self, embed_dim, nhead, num_layers=2, win_size=12, shift=True,
                 cwi=False, use_efficient_local_global_mixer=False,
                 elgm_channel_ratio=0.5, elgm_replace_layer=1,
                 elgm_max_residual=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = embed_dim
        self.num_layers = num_layers
        self.win_size = win_size
        
        self.sla_layers = list()
        
        if use_efficient_local_global_mixer and not (
                0 <= elgm_replace_layer < num_layers):
            raise ValueError('ELGM_REPLACE_LAYER must index an existing SlotFormer layer')

        shift = [False, shift]
        for i in range(num_layers):
            if use_efficient_local_global_mixer and i == elgm_replace_layer:
                self.sla_layers.append(ELGMLayer(
                    embed_dim, directions=2, win_size=win_size,
                    shift=shift[i], cwi=cwi,
                    channel_ratio=elgm_channel_ratio,
                    max_residual=elgm_max_residual
                ))
            else:
                self.sla_layers.append(SFLayer(
                    embed_dim, nhead=nhead, directions=2,
                    win_size=win_size, shift=shift[i], cwi=cwi
                ))
            
        self.sla_layers = nn.ModuleList(self.sla_layers)
        
    def forward(self, src):
        
        for idx, enc in enumerate(self.sla_layers):
            src = enc(src)

        return src
#end
    
    
class SFLayer(nn.Module):
    def __init__(self, embed_dim, nhead, directions=2, win_size=12, shift=False, cwi=False):
        super().__init__()
        self.nhead = nhead
        self.d_model = embed_dim
        self.directions = directions
        self.win_size = win_size
        self.shift = shift
        self.self_norm_layers = list()
        self.self_attn_layers = list()
        self.ffn_layers = list()
        
        for dir in range(directions):
            self.self_norm_layers.append( nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01) )
            self.self_attn_layers.append( SelfAttnLayer(embed_dim, nhead) )
            self.ffn_layers.append( CWI_FFN_Layer(embed_dim, conv_size=self.win_size+1) if cwi 
                                    else FFN_Layer(embed_dim, conv_size=self.win_size+1))
        
        self.self_norm_layers = nn.ModuleList(self.self_norm_layers)
        self.self_attn_layers = nn.ModuleList(self.self_attn_layers)
        self.ffn_layers = nn.ModuleList(self.ffn_layers)
    
    def get_direction_attrs(self, src, direction=0):
        if self.shift:
            batch_win_coords = torch.cat([ src.indices[:, :1], (src.indices[:, [direction+1]] + self.win_size // 2) // self.win_size], dim=1) 
        else:
            batch_win_coords = torch.cat([ src.indices[:, :1], src.indices[:, [direction+1]] // self.win_size ], dim=1) 
    
        _, batch_win_inds = torch.unique(batch_win_coords, return_inverse=True, dim=0)
        
        batch_win_inds, perm = torch.sort(batch_win_inds, dim=0)
    
        counts = torch.bincount(batch_win_inds)
        offsets = F.pad( torch.cumsum(counts, dim=0), (1,0), mode='constant', value=0)[:-1]
        return counts, offsets, batch_win_inds, perm

    def forward(self, src):

        for i in range(self.directions):
            counts, offsets, win_inds, perm = self.get_direction_attrs(src, i)
            norm_features = self.self_norm_layers[i](src.features[perm])
            #改进16：仅训练且需梯度时做激活检查点；验证直接前向，避免no_grad告警且不改变训练路径。
            if self.training and torch.is_grad_enabled() and norm_features.requires_grad:
                x = checkpoint(self.self_attn_layers[i], norm_features, offsets, counts, win_inds)
            else:
                x = self.self_attn_layers[i](norm_features, offsets, counts, win_inds)
            #end
            inv_x = x.clone()
            inv_x[perm] = x
            src = replace_feature(src, src.features + inv_x)

            src = self.ffn_layers[i](src)
            
        return src


class SelfAttnLayer(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads    
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x, offsets, counts, batch_win_inds, eps=1e-3):
        with autocast(enabled=True, dtype=torch.float32):
            N, C = x.shape
            
            qkv = self.qkv(x).reshape(N, 3, C)

            q, k, v = qkv.unbind(1)

            q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])

            q = F.relu(q)
            k = F.relu(k)

            kv = scatter_matmul_kv(k, v, offsets, counts)
            s = torch_scatter.scatter_add(k, batch_win_inds, dim=0)
        
            y = scatter_matmul_qc(q, kv, offsets, counts)
            z = torch.sum(s[batch_win_inds, ...] * q, -1, keepdim=True)
        
            y = y / (z + eps)

            y = rearrange(y, "n h c -> n (h c)", h=self.num_heads)
            
            y = self.proj(y)
    
        return y


#改进3：ELGM以局部SubMConv和按slot线性均值聚合替代一层注意力，并以零初始化有界残差稳定训练。
class ELGMLayer(nn.Module):
    def __init__(self, embed_dim, directions=2, win_size=12, shift=False,
                 cwi=False, channel_ratio=0.5, max_residual=0.1):
        super().__init__()
        if not 0.0 < channel_ratio < 1.0:
            raise ValueError('ELGM_CHANNEL_RATIO must be in (0, 1)')
        if max_residual <= 0:
            raise ValueError('ELGM_MAX_RESIDUAL must be positive')

        self.directions = directions
        self.win_size = win_size
        self.shift = shift
        self.max_residual = max_residual
        self.local_dim = max(
            1, min(embed_dim - 1, int(round(embed_dim * channel_ratio)))
        )
        self.global_dim = embed_dim - self.local_dim

        self.norm_layers = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        self.global_projections = nn.ModuleList()
        self.output_projections = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for direction in range(directions):
            self.norm_layers.append(
                nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
            )
            self.local_convs.append(spconv.SubMConv2d(
                self.local_dim, self.local_dim, kernel_size=3,
                stride=1, padding=1, bias=False,
                indice_key='elgm_local_%d' % direction
            ))
            self.global_projections.append(
                nn.Linear(self.global_dim, self.global_dim, bias=False)
            )
            self.output_projections.append(
                nn.Linear(embed_dim, embed_dim, bias=False)
            )
            self.gate_layers.append(nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid(),
            ))
            self.ffn_layers.append(
                CWI_FFN_Layer(embed_dim, conv_size=self.win_size + 1)
                if cwi else FFN_Layer(
                    embed_dim, conv_size=self.win_size + 1
                )
            )

            nn.init.zeros_(self.output_projections[-1].weight)
            nn.init.zeros_(self.gate_layers[-1][0].weight)
            nn.init.zeros_(self.gate_layers[-1][0].bias)

    def get_slot_mean(self, src, features, direction):
        coordinate = src.indices[:, direction + 1:direction + 2]
        if self.shift:
            coordinate = coordinate + self.win_size // 2
        slot_coordinates = torch.cat([
            src.indices[:, :1], coordinate // self.win_size
        ], dim=1)
        unique_slots, inverse = torch.unique(
            slot_coordinates, dim=0, return_inverse=True
        )
        if unique_slots.shape[0] == 0:
            return torch.zeros_like(features)

        slot_sums = features.new_zeros(
            (unique_slots.shape[0], features.shape[1])
        )
        slot_sums.index_add_(0, inverse, features)
        slot_counts = features.new_zeros((unique_slots.shape[0], 1))
        slot_counts.index_add_(
            0, inverse, features.new_ones((features.shape[0], 1))
        )
        return slot_sums[inverse] / slot_counts[inverse].clamp_min(1.0)

    def forward(self, src):
        for direction in range(self.directions):
            normalized = self.norm_layers[direction](src.features)
            local_tensor = replace_feature(
                src, normalized[:, :self.local_dim].contiguous()
            )
            local_features = self.local_convs[direction](local_tensor).features
            global_features = normalized[:, self.local_dim:].contiguous()

            with autocast(enabled=False):
                global_context = self.get_slot_mean(
                    src, global_features.float(), direction
                )
                global_context = self.global_projections[direction](
                    global_context
                )
                mixed = self.output_projections[direction](torch.cat([
                    local_features.float(), global_context
                ], dim=1))
                gate = self.gate_layers[direction](normalized.float())
                mixed = (
                    self.max_residual * gate * torch.tanh(mixed)
                ).to(src.features.dtype)

            src = replace_feature(src, src.features + mixed)
            src = self.ffn_layers[direction](src)

        return src
#end
    

class CWI_FFN_Layer(spconv.SparseModule):
    
    def __init__(self, embed_dim, indice_key='former', conv_size=13):
        super(CWI_FFN_Layer, self).__init__()

        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        
        self.conv_k = spconv.SubMConv2d(
            embed_dim // 4, embed_dim // 4,kernel_size=3, stride=1, padding=1, indice_key=indice_key + 'k'
        )
        self.conv_h = spconv.SubMConv2d(
            embed_dim // 4, embed_dim // 4,kernel_size=(1, conv_size), stride=(1, 1), padding=(0, conv_size//2), indice_key=indice_key + 'h'
        )
        self.conv_w = spconv.SubMConv2d(
            embed_dim // 4, embed_dim // 4,kernel_size=(conv_size, 1), stride=(1, 1), padding=(conv_size//2, 0), indice_key=indice_key + 'w'
        )

        self.bn2 = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(embed_dim, embed_dim*2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim*2, embed_dim)
        
        self.group_dim = embed_dim // 4

    def forward(self, src):

        src = replace_feature(src, self.bn(src.features))
        src_k = replace_feature(src, src.features[:, :self.group_dim])
        src_h = replace_feature(src, src.features[:, self.group_dim:2*self.group_dim])
        src_w = replace_feature(src, src.features[:, 2*self.group_dim:3*self.group_dim])
        
        src_k = self.conv_k(src_k).features
        src_h = self.conv_h(src_h).features
        src_w = self.conv_w(src_w).features
        src_i = src.features[:, 3*self.group_dim:]
        src2 = replace_feature(src, torch.cat([src_k, src_h, src_w, src_i], 1))
        src = replace_feature(src, src.features + src2.features)

        src2 = replace_feature(src2, self.fc2(self.act(self.fc1(self.bn2(src.features)))))
        src = replace_feature(src, src.features + src2.features)
        
        return src


class FFN_Layer(nn.Module):
    
    def __init__(self, embed_dim, **kwargs):
        super(FFN_Layer, self).__init__()
        
        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(embed_dim, embed_dim*2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim*2, embed_dim)
    
    def forward(self, src):

        ff_features = self.fc2(self.act(self.fc1(self.bn(src.features))))
        src = replace_feature(src, src.features + ff_features)
       
        return src
