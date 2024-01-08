import pytorch_lightning as pl
import torch
from torch import nn

from src.lib import conv_norm_act, SoftPositionEmbed, init_module

class CNNEncoder(pl.LightningModule):
    def __init__(self, norm, ks, channels, resolution, out_size, add_pos_emb, strides, flatten_output):
        super().__init__()

        self.resolution = resolution
        self.channels = channels
        self.ks = ks
        self.norm = norm
        self.out_size = out_size
        self.add_pos_emb = add_pos_emb
        self.strides = strides
        self.flatten_output = flatten_output

        # Build Encoder
        # Conv CNN --> PosEnc --> MLP
        enc_layers = len(self.channels) - 1
        self.encoder = nn.Sequential(*[
            conv_norm_act(
                self.channels[i],
                self.channels[i + 1],
                kernel_size=self.ks,
                stride=self.strides[i],
                norm=self.norm,
                act='relu' if i != (enc_layers - 1) else '')
            for i in range(enc_layers)
        ])

        out_resolution = self.resolution
        for i in range(enc_layers):
            out_resolution = (int((out_resolution[0] - self.ks + 2 * (self.ks // 2)) / self.strides[i]) + 1,
                              int((out_resolution[1] - self.ks + 2 * (self.ks // 2)) / self.strides[i]) + 1)

        if self.add_pos_emb:
            self.encoder_pos_embedding = SoftPositionEmbed(self.channels[-1],
                                                           out_resolution)
        else:
            self.encoder_pos_embedding = None

        out_layer_in_size = self.channels[-1] * out_resolution[0] * out_resolution[1] if self.flatten_output else self.channels[-1]

        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(out_layer_in_size),
            nn.Linear(out_layer_in_size, self.out_size),
            nn.ReLU(),
            nn.Linear(self.out_size, self.out_size),
        )



    def forward(self, x):
        out = self.encoder(x)

        if self.encoder_pos_embedding is not None:
            out = self.encoder_pos_embedding(out)

        if self.flatten_output:
            out = torch.flatten(out, start_dim=1)
        else:
            out = torch.flatten(out, start_dim=2, end_dim=3)
            out = out.permute(0, 2, 1).contiguous()

        out = self.encoder_out_layer(out)

        return out
