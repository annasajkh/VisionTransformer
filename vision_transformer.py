!pip install einops

import torch
import torch.nn as nn

from torchvision.transforms import Compose, Resize, ToTensor
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn import functional as F
from tqdm import tqdm
from torch.optim import AdamW
from IPython.display import clear_output
from torch.optim import AdamW

#GLU Variant https://arxiv.org/abs/2002.05202
#SwiGLU https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


#NormFormer https://arxiv.org/abs/2110.09456
class TransformerBlock(nn.Module): 
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_scale=4,
        attn_mask=None,
        attn_drop=0,
        resid_pdrop=0
    ):
        super().__init__()
        self.attn_mask = attn_mask
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, bias=False)
        
        self.pre_attn_layer_norm = nn.LayerNorm(d_model)
        self.pre_mlp_layer_norm = nn.LayerNorm(d_model)
        self.post_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_scale * 2, bias=False),
            SwiGLU(),
            nn.LayerNorm(d_model * mlp_scale),
            nn.Linear(d_model * mlp_scale, d_model, bias=False),
            nn.Dropout(resid_pdrop)
        )

    def attention(self, x, cache=None): 
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        
        return self.attn(x, x, x, attn_mask=self.attn_mask, need_weights=False)[0]
    
    def forward(self, x):
        x = x + self.post_attn_layer_norm(self.attention(self.pre_attn_layer_norm(x)))
        x = x + self.mlp(self.pre_mlp_layer_norm(x))
        
        return x


class Transformer(nn.Module): 
    def __init__(
        self,
        n_context,
        n_embed,
        n_heads,
        n_layers, 
        mlp_scale=4,
        attn_mask=None,
        attn_drop=0,
        resid_pdrop=0,
        embed_pdrop=0
    ):
        super().__init__()
        
        self.drop = nn.Dropout(embed_pdrop) 
        self.pos_embed = nn.Parameter(torch.randn(1, n_context, n_embed)) 
        self.layers = nn.Sequential(*[TransformerBlock(n_embed,
                                                       n_heads,
                                                       attn_mask=attn_mask,
                                                       mlp_scale=mlp_scale,
                                                       attn_drop=attn_drop,
                                                       resid_pdrop=resid_pdrop) for _ in range(n_layers)]) 
        self.ln_pre = nn.LayerNorm(n_embed)
        
    #the input is an embbeding with this shape (batch, n_context, n_embed) 
    def forward(self, x):
        x = self.drop(x + self.pos_embed[:, :x.shape[1], :])
        x = self.ln_pre(x) 

        x = x.permute(1, 0, 2)
        x = self.layers(x)
        x = x.permute(1, 0, 2)
        
        return x

#modified version of https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632 
#and https://github.com/openai/CLIP/blob/main/clip/model.py 
#vision transformer is just transformer encoder with conv layer to project the image
class VisionTransformer(nn.Module):
    def __init__(self, n_heads, n_layers, in_channels, patch_size, n_embed, img_size, n_class=None, mlp_scale=4, masked=False):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, n_embed, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embed))
        self.transformer = Transformer((img_size // patch_size) ** 2 + 1, 
                                       n_embed=n_embed,
                                       n_heads=n_heads,
                                       n_layers=n_layers,
                                       mlp_scale=mlp_scale)
        
        self.ln_post = nn.LayerNorm(n_embed)
        self.reduce = Reduce("b n e -> b e", reduction="mean")

        self.out_head = nn.Linear(n_embed, n_class) if n_class is not None else None
        self.n_class = n_class
    
    def forward(self,x): 
        b, _, _, _ = x.shape

        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)

        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        x = self.ln_post(self.reduce(x))

        if self.n_class != None:
            x = self.out_head(x)

        return x


img = Image.open("img.png")

transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionTransformer(n_heads=6, n_layers=6, in_channels=3, patch_size=16, n_embed=768, img_size=224)
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=3e-4)

print(model(x.to(device)).shape)
