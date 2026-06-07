import torch.nn as nn
import torch
from itertools import repeat
import collections.abc

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        
    def forward(self, point, feature,pos,posk):
        padded_feat = point
        
        def pos_emb(feat,pos):
            A,B,C = pos.shape
            new_feat = torch.zeros_like(feat)
            new_feat[:,:B,:] = feat[:,:B,:] + pos
            new_feat[:,B:,:] = feat[:,B:,:]
            return new_feat
        
        # Cross-attention with external feature
        feature = self.norm_y(feature)  # [B, L, C]
        padded_feat = padded_feat + self.drop_path(
            self.cross_attn(
                query=pos_emb(self.norm2(padded_feat),pos),
                key=pos_emb(feature,posk),
                value=feature
            )[0]
        )

        return padded_feat
    
class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, num_cond,mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, bias=qkv_bias, dropout=attn_drop, batch_first=True)
        self.cross_attns = nn.ModuleList()
        for _ in range(num_cond):
            self.cross_attns.append(CrossAttention(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                 drop_path, act_layer, norm_layer, norm_mem))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, point, features,pos,posk):
        padded_feat = point
        
        def pos_emb(feat,pos):
            A,B,C = pos.shape
            new_feat = torch.zeros_like(feat)
            new_feat[:,:B,:] = feat[:,:B,:] + pos
            new_feat[:,B:,:] = feat[:,B:,:]
            return new_feat
        
        # Self-attention
        normed_feat = self.norm1(padded_feat)
        padded_feat = padded_feat + self.drop_path(
            self.attn(
                query=pos_emb(normed_feat,pos),
                key=pos_emb(normed_feat,pos),
                value=normed_feat,
            )[0]
        )
        
        for cross_attn, feature in zip(self.cross_attns,features):
            padded_feat = cross_attn(padded_feat,feature,pos,posk)
        
        padded_feat = padded_feat + self.drop_path(
            self.mlp(self.norm3(padded_feat))
        )

        return padded_feat
    
class Fussion(nn.Module):

    def __init__(self, depth=2,dim=256, num_heads=8, num_cond=3,mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.dec1 = nn.ModuleList()
        self.depth = depth
        mlp_hidden_dim = int(dim)
        self.iou_head = nn.ModuleList()
        self.lang_head = nn.ModuleList()
        self.mask_heads = nn.ModuleList()
        self.confidence_head = nn.ModuleList()
        for _ in range(self.depth):
            self.dec1.append(DecoderBlock(dim=dim, num_heads=8, num_cond=num_cond))
            self.iou_head.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=1,act_layer=act_layer, drop=drop))
            self.lang_head.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=37,act_layer=act_layer, drop=drop))
            self.mask_heads.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=256,act_layer=act_layer, drop=drop))

    def forward(self,query,all_view,emb1,emb2,posq=None,all_pos=None):
        def get_mask(query,emb,i):
            weights = torch.zeros_like(query)
            weights += self.mask_heads[-1](query)
            return torch.einsum("bqc,bchw->bqhw",weights,emb)
        all_iou = []
        all_mask1 = []
        all_mask2 = []
        all_lang = []
        for ind in range(self.depth):
            query = self.dec1[ind](query,all_view[ind],posq,all_pos[ind])
            obj = query
            iou_scores = self.iou_head[-1](obj)
            lang_scores = self.lang_head[-1](obj)
            mask1 = get_mask(obj,emb1,ind)
            mask2 = get_mask(obj,emb2,ind)
            if len(all_mask1) > 0:
                mask1+=all_mask1[-1]
            if len(all_mask2) > 0:
                mask2+=all_mask2[-1]
            all_iou.append(iou_scores)
            all_lang.append(lang_scores)
            all_mask1.append(mask1)
            all_mask2.append(mask2)
        
        
        return (obj,all_iou,all_lang),all_mask1,all_mask2

class Fussionx(nn.Module):

    def __init__(self, depth=2,dim=256, num_heads=8, num_cond=3,mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.dec1 = nn.ModuleList()
        # self.dec2 = nn.ModuleList()
        self.depth = depth
        mlp_hidden_dim = int(dim)
        self.iou_head = nn.ModuleList()
        # self.exist_head = nn.ModuleList()
        self.lang_head = nn.ModuleList()
        self.mask_heads = nn.ModuleList()
        self.confidence_head = nn.ModuleList()
        for _ in range(self.depth):
            self.dec1.append(DecoderBlock(dim=dim, num_heads=8, num_cond=num_cond))
            self.iou_head.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=1,act_layer=act_layer, drop=drop))
            # self.exist_head.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=40,act_layer=act_layer, drop=drop))
            self.lang_head.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=37,act_layer=act_layer, drop=drop))
            self.mask_heads.append(Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=256,act_layer=act_layer, drop=drop))

    def forward(self,query,all_view,emb1,emb2,posq=None,all_pos=None):
        '''
        obj_query: B N+iou+exist 256
        nopo: B 128*128 256
        sam: B 64*64 256
        clip: B 9 256
        emb1: B 256 256 256
        '''
        def get_mask(query,emb,i):
            weights = torch.zeros_like(query)
            weights += self.mask_heads[-1](query)
            return torch.einsum("bqc,bchw->bqhw",weights,emb)
        all_iou = []
        # all_exist = []
        # all_lang = []
        all_mask1 = []
        all_mask2 = []
        all_lang = []
        for ind in range(self.depth):
            query = self.dec1[ind](query,all_view[ind],posq,all_pos[ind])
            obj = query
            iou_scores = self.iou_head[ind](obj)
            # exist_scores = self.exist_head[ind](exist)
            lang_scores = self.lang_head[ind](obj)
            mask1 = get_mask(obj,emb1,ind)
            mask2 = get_mask(obj,emb2,ind)
            if len(all_mask1) > 0:
                mask1+=all_mask1[-1]
            if len(all_mask2) > 0:
                mask2+=all_mask2[-1]
            all_iou.append(iou_scores)
            # all_exist.append(exist_scores)
            all_lang.append(lang_scores)
            all_mask1.append(mask1)
            all_mask2.append(mask2)
        
        return (obj,all_iou,all_lang),all_mask1,all_mask2
    