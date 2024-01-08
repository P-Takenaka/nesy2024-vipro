import pytorch_lightning as pl
import ml_collections
from torch import nn
import torch
from torch.nn import functional as F

from src.lib import init_module

class ProcModule(pl.LightningModule):
    def __init__(self, F_module, P_in, P_out, F_in, F_out, z_a_size, z_b_size, z_c_size,
                 latent_size=None, D_module=None, has_temporal_dim=True, num_objects=None):
        super().__init__()

        self.z_a_size = z_a_size
        self.z_b_size = z_b_size
        self.z_c_size = z_c_size

        # Whether the output should have a temporal dim
        self.has_temporal_dim = has_temporal_dim
        # If this is set, we assume there is no object dimension (no object centric model)
        self.num_objects = num_objects

        self.latent_size = latent_size

        self.Fs = nn.ModuleList([init_module(
            F_module)])
        self.D = init_module(D_module) if D_module is not None else None

        if P_in is None:
            self.P_in = None
        elif type(P_in) == dict or type(P_in) == ml_collections.ConfigDict:
            self.P_in = init_module(P_in, input_size=self.latent_size if self.latent_size else self.z_a_size + self.z_b_size + self.z_c_size, output_size=self.z_a_size + self.z_b_size + self.z_c_size)
        else:
            self.P_in = P_in

        if P_out is None:
            self.P_out = None
        elif type(P_out) == dict or type(P_out) == ml_collections.ConfigDict:
            self.P_out = init_module(P_out, input_size=self.z_a_size + self.z_b_size + self.z_c_size, output_size=self.latent_size if self.latent_size else self.z_a_size + self.z_b_size + self.z_c_size)
        else:
            self.P_out = P_out

        self.F_in = init_module(F_in, input_size=self.z_a_size, output_size=self.Fs[0].get_sym_state_size() * (self.num_objects if self.num_objects is not None else 1))
        self.F_out = init_module(F_out, input_size=self.Fs[0].get_sym_state_size() * (self.num_objects if self.num_objects is not None else 1), output_size=self.z_a_size)

    def read_sym_state(self, z):
        z = self.F_in(z[..., :self.z_a_size])

        return self.Fs[0].convert_tensor_to_state_dict(
            z, has_object_dim=self.num_objects is None)

    def split_z(self, z):
        return torch.split(z, split_size_or_sections=[self.z_a_size, self.z_b_size, self.z_c_size], dim=-1)

    def forward(self, z, z_a_sym_dict, frame_idx, dataloader_idx=0, **kwargs):
        assert((z.shape[-1] == (self.z_a_size + self.z_b_size + self.z_c_size)))
        assert(len(z.shape) == 4 if self.num_objects is None and self.has_temporal_dim else True)
        assert(len(z.shape) == 3 if self.num_objects is None and not self.has_temporal_dim else True)
        assert(len(z.shape) == 3 if self.num_objects is not None and self.has_temporal_dim else True)
        assert(len(z.shape) == 2 if self.num_objects is not None and not self.has_temporal_dim else True)

        z_a_sym_dict = self.Fs[dataloader_idx](
            z_a_sym_dict, frame_idx=frame_idx)

        z_a_sym = self.Fs[dataloader_idx].convert_state_dict_to_tensor(
            z_a_sym_dict, keep_object_dim=self.num_objects is None)

        if self.D is not None:
            z_b = self.D(z, **kwargs)
            if self.has_temporal_dim:
                assert(len(z_b.shape) == 4 if self.num_objects is None else len(z_b.shape) == 3)
                z_b = z_b[:, -1:]
            else:
                assert(len(z_b.shape) == 3 if self.num_objects is None else len(z_b.shape) == 2)
        else:
            assert(not self.z_b_size)
            z_b = None

        z_a = self.F_out(z_a_sym)

        assert(len(z_a.shape) == 2 if self.num_objects else len(z_a.shape) == 3)
        if self.has_temporal_dim:
            z_a = torch.unsqueeze(z_a, dim=1)

        if z_b is None:
            if self.z_c_size:
                z_c = z[:, -1:, ..., -self.z_c_size:] if self.has_temporal_dim else z[..., -self.z_c_size:]

                z = torch.cat([z_a, z_c], dim=-1)
            else:
                z = z_a
        else:
            if self.z_c_size:
                z_c = z[:, -1:, ..., -self.z_c_size:] if self.has_temporal_dim else z[..., -self.z_c_size:]

                z = torch.cat([z_a, z_b, z_c], dim=-1)
            else:
                z = torch.cat([z_a, z_b], dim=-1)

        if self.has_temporal_dim:
            decoded_state = self.read_sym_state(z[:, -1])
        else:
            decoded_state = self.read_sym_state(z)

        return z, z_a_sym_dict, decoded_state
