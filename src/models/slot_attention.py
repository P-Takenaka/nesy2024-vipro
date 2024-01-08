import pytorch_lightning as pl

import torch

import torch
from torch import nn
from torch.nn import functional as F

# Code adapted from https://github.com/pairlab/SlotFormer/blob/master/slotformer/base_slots/models/savi.py

class SlotAttention(pl.LightningModule):
    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        if self.mlp_hidden_size:
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.slot_size),
                nn.Linear(self.slot_size, self.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_size, self.slot_size),
            )
        else:
            self.mlp = None

    def forward(self, inputs, slots, dataloader_idx=0):
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.

        k = self.project_k(inputs)

        v = self.project_v(inputs)


        assert len(slots.shape) == 3

        for _ in range(self.num_iterations):
            slots_prev = slots

            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)

            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)

            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.flatten(0, 1),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            if self.mlp is not None:
                slots = slots + self.mlp(slots)

        return slots

