import torch

from torch import nn

from torchvision.transforms import Compose, Resize, ToTensor
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

#modified version of https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632 
#and https://github.com/openai/CLIP/blob/main/clip/model.py 
#vision transformer is just transformer encoder with cony layer to proccess images 
class VisionTransformer(nn.Module):
  def __init__(self, n_heads, n_layers, in_channels, patch_size, n_embed, img_size, n_class=None, mlp_scale=4):
    super().__init__()
    
    self.projection = nn.Sequential(
      nn.Conv2d(in_channels, n_embed, kernel_size=patch_size, stride=patch_size),
      Rearrange("b e (h) (w) -> b (h w) e"),
    )
    
    self.cls_token = nn.Parameter(torch.randn(1, 1, n_embed))
    self.transformer = Transformer((img_size // patch_size) ** 2 + 1, n_embed=n_embed, n_heads=n_heads, n_layers=n_layers, mlp_scale=mlp_scale)
    
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
    x = self.in_post(self.reduce(x))

    if self.n_class != None:
      x = self.out_head(x)
    
    return x
