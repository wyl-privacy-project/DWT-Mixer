import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from DWT import DWT1DForward,DWT1DInverse
from pytorch_wavelets import DWT1DForward, DWT1DInverse
class DWT_MIXER(nn.Module):

    def __init__(self, num_mixers: int, max_seq_len: int, hidden_dim: int, mlp_hidden_dim: int, **kwargs):
        super(DWT_MIXER, self).__init__(**kwargs)
        self.hidden_dim=hidden_dim
        self.max_len=max_seq_len
        self.mixers = nn.Sequential(*[
            MixerLayer(max_seq_len, hidden_dim, mlp_hidden_dim, mlp_hidden_dim) for _ in range(num_mixers)
        ])


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mixers(inputs)

class MixerLayer(nn.Module):

    def __init__(self, max_seq_len: int, hidden_dim: int, channel_hidden_dim: int, seq_hidden_dim: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.FC1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # self.FC2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim, bias=True),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        # )
        self.FC3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # self.FC4 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim, bias=True),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        # )
        self.xfm = DWT1DForward(J=1, mode='zero', wave='haar')
        self.ifm = DWT1DInverse(mode='zero', wave='haar')
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_2 = MlpLayer(hidden_dim, channel_hidden_dim)

        self.scale = hidden_dim ** -0.5

    # def forward(self, x):
    #     input = x
    #     B, N, D = input.shape
    #     x1 = self.FC1(input)
    #     x_dwt0, x_dwt1 = self.xfm(x1.permute(0, 2, 1))  # B D N/2
    #     x_shift1 = torch.roll(x1, (0, 0, -1), (1, 1, 1)) # B N D
    #     x_dwt3, x_dwt4 = self.xfm(x_shift1.permute(0, 2, 1))  # B D N/2
    #     # xx =x_dwt0*x_dwt1[0]+x_dwt0
    #     # xxx=x_dwt3*x_dwt4[0]+x_dwt3
    #     resu_loc = torch.stack([x_dwt1[0].permute(0, 2, 1), x_dwt4[0].permute(0, 2, 1)], 2).reshape(-1, N, D)
    #     x_local=x1*resu_loc
    #     # cc=self.ifm(cc)
    #     # cc=self.FC2(cc.permute(0,2,1))
    #     # dd=self.ifm(dd)
    #     # dd=self.FC3(dd.permute(0,2,1))
    #     # resu =cc+dd
    #     # resuu = self.FC4(resu.permute(0,2,1))
    #     # x_dwt00, x_dwt11 = self.xfm(resuu.permute(0, 2, 1))  # B D N/2
    #     # x_shift2 = torch.roll(resuu, (0, 0, -1), (1, 1, 1))  # B N D
    #     # x_dwt33, x_dwt44 = self.xfm(x_shift2.permute(0, 2, 1))  # B D N/2
    #     # x_resu22 = torch.stack([x_dwt00.permute(0, 2, 1), x_dwt33.permute(0, 2, 1)], 2).reshape(-1, N, D)  # B N D
    #     # resu2 = self.FC2(x_resu22)
    #     # resu = torch.stack([xx.permute(0, 2, 1), xxx.permute(0, 2, 1)], 2).reshape(-1, N, D)
    #     resu_ful = torch.stack([x_dwt0.permute(0, 2, 1), x_dwt3.permute(0, 2, 1)], 2).reshape(-1, N, D)
    #     # resu = self.FC3(resu)
    #     resu = self.FC3(resu_ful)
    #     # resu = self.layer_norm(resu)
    #     # x_resu2 = torch.stack([xx.permute(0, 2, 1), xxx.permute(0, 2, 1)], 2).reshape(-1, N, D)  # B N D
    #
    #
    #     redus = resu+x1+x_local
    #     residual = redus
    #     outputs = self.layer_norm(residual)
    #     outputs = self.mlp_2(outputs) + residual
    #
    #     return outputs
    #
    #
    #     # self.FC1 = nn.Sequential(
    #     #     nn.Linear(hidden_dim, hidden_dim, bias=True),
    #     #     nn.LayerNorm(hidden_dim),
    #     #     nn.GELU(),
    #     # )
    #     # self.FC3 = nn.Sequential(
    #     #     nn.Linear(hidden_dim, hidden_dim , bias=True),
    #     #     nn.LayerNorm(hidden_dim ),
    #     #     nn.GELU(),
    #     # )
    #     # self.xfm = DWT1DForward(J=1, mode='zero', wave='haar')
    #     # self.ifm = DWT1DInverse(mode='zero', wave='haar')
    #     # self.layer_norm = nn.LayerNorm(hidden_dim)
    #     # self.mlp_2 = MlpLayer(hidden_dim, channel_hidden_dim)
    #     #
    #     # self.scale=hidden_dim**-0.5
    # √ √ √
    def forward(self, x):
        input =x
        B,N,D=input.shape
        x1 = self.FC1(input)
        x_dwt0,x_dwt1 = self.xfm(x1.permute(0,2,1))
        x_shift1 = torch.roll(x1, (0, 0, -1), (1, 1, 1))
        # x_shift2 = torch.roll(x1, (0, 0, 1), (1, 1, 1))
        x_shift2 = torch.roll(x1, (0, 0, 1), (1, 1, 1))
        # x_resu = x1+x_shift1
        x_dwt3, x_dwt4 = self.xfm(x_shift1.permute(0, 2, 1))
        x_dwt5, x_dwt6 = self.xfm(x_shift2.permute(0, 2, 1))
        # x_dwt5, x_dwt6 = self.xfm(x_shift1.permute(0, 2, 1))
        x_resu1=torch.stack([x_dwt0.permute(0,2,1), x_dwt3.permute(0,2,1)], 2).reshape(-1, N, D)
        x_resu2=torch.stack([x_dwt5.permute(0,2,1), x_dwt0.permute(0,2,1)], 2).reshape(-1, N, D)
        resu =self.FC3(x_resu1+x_resu2)
        redus = resu + x1
        residual = redus
        outputs = self.layer_norm(residual)
        outputs = self.mlp_2(outputs) + residual
        return outputs

    # × × √
    # def forward(self, x):
    #     input =x
    #     B,N,D=input.shape
    #     outputs = self.layer_norm(x)
    #     outputs = self.mlp_2(outputs) + x
    #     return outputs

    #x √ √
    # def forward(self, x):
    #     input =x
    #     B,N,D=input.shape
    #     x_dwt0,x_dwt1 = self.xfm(x.permute(0,2,1))
    #     x_shift1 = torch.roll(x, (0, 0, -1), (1, 1, 1))
    #     x_dwt3, x_dwt4 = self.xfm(x_shift1.permute(0, 2, 1))
    #     x_resu=torch.stack([x_dwt0.permute(0,2,1), x_dwt3.permute(0,2,1)], 2).reshape(-1, N, D)
    #     redus = x_resu + x
    #     outputs = self.layer_norm(redus)
    #     outputs = self.mlp_2(outputs) + redus
    #     return outputs

    # √ x  √
    # def forward(self, x):
    #     input = x
    #     B, N, D = input.shape
    #     x1 = self.FC1(input)
    #     resu = self.FC3(x1)
    #     redus = resu + x1
    #     outputs = self.layer_norm(redus)
    #     outputs = self.mlp_2(outputs) + redus
    #     return outputs

class MlpLayer(nn.Module):

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim)
        ])
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)

