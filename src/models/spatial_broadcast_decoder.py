import pytorch_lightning as pl

from torch import nn

from src.lib import deconv_norm_act, SoftPositionEmbed

class SpatialBroadcastDecoder(pl.LightningModule):
    def __init__(self, input_size, resolution, channels, ks, norm, strides):
        super().__init__()

        self.input_size = input_size
        self.resolution = resolution
        self.channels = channels
        self.ks = ks
        self.norm = norm
        self.strides = strides

        modules = []
        for i in range(len(self.channels) - 2):
            modules.append(
                deconv_norm_act(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=self.ks,
                    stride=strides[i],
                    norm=self.norm,
                    act='relu'))

        modules.append(nn.Conv2d(
                self.channels[-2], self.channels[-1], kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.input_size,
                                                       self.resolution)

    def forward(self, x):
        assert(len(x.shape) == 2)
        out = x.view(x.shape[0], x.shape[1], 1, 1)
        out = out.repeat(1, 1, self.resolution[0], self.resolution[1])

        out = self.decoder_pos_embedding(out)
        out = self.decoder(out)

        return out
