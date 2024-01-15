import torch
import torch.nn as nn
import os
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange

def inputs_deal(inputs):
    return inputs if isinstance(inputs, tuple) else(inputs, inputs)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N =query.shape[0]
        value_len , key_len , query_len = values.shape[1], keys.shape[1], query.shape[1]
        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        attention = torch.softmax(energy/ (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, x, type_mode):
        if type_mode == 'original':
            attention = self.attention(value, key, query)
            x = self.dropout(self.norm(attention + x))
            forward = self.feed_forward(x)
            out = self.dropout(self.norm(forward + x))
            return out
        else:
            attention = self.attention(self.norm(value), self.norm(key), self.norm(query))
            x =self.dropout(attention + x)
            forward = self.feed_forward(self.norm(x))
            out = self.dropout(forward + x)
            return out

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout = 0,
            type_mode = 'original'
        ):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.type_mode = type_mode
        self.Query_Key_Value = nn.Linear(embed_size, embed_size * 3, bias = False)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    )
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            QKV_list = self.Query_Key_Value(x).chunk(3, dim = -1)
            x = layer(QKV_list[0], QKV_list[1], QKV_list[2], x, self.type_mode)
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                image_size,
                patch_size,
                num_classes,
                embed_size,
                num_layers,
                heads,
                mlp_dim,
                pool = 'cls',
                channels = 3,
                dropout = 0,
                emb_dropout = 0.1,
                type_mode = 'vit'):
        super(VisionTransformer, self).__init__()
        img_h, img_w = inputs_deal(image_size)
        patch_h, patch_w = inputs_deal(patch_size)
        assert img_h % patch_h == 0 and img_w % patch_w == 0, 'Img dimensions can be divisible by the patch dimensions'
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        patch_size = channels * patch_h * patch_w
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_h, p2=patch_w),
            nn.Linear(patch_size, embed_size, bias=False)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerEncoder(embed_size,
                                    num_layers,
                                    heads,
                                    mlp_dim,
                                    dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d ->b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    vit = VisionTransformer(
            image_size = 256,
            patch_size = 16,
            num_classes = 10,
            embed_size = 256,
            num_layers = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    img = torch.randn(3, 3, 256, 256)
    pred = vit(img)
    print(pred)

