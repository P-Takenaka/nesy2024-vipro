import pytorch_lightning as pl

import torch.optim as optim
import torch

import torch
from torch import nn
from torch.nn import functional as F

from src.lib import init_module, concat_dict, detach_dict, elup1

def get_output_slice(out_dict, i):
    result = {}
    for k, v in out_dict.items():
        if k == 'loss' or k == 'batch_idx' or k == 'loss_dict':
            continue

        if type(v) == dict:
            result[k] = get_output_slice(v, i)
        else:
            result[k] = v[:, i:i+1] if v is not None else v

    return result

def slot_decode(z, decoder):
    assert(len(z.shape) == 3)
    bs, num_slots, slot_size = z.shape

    z = z.view(bs * num_slots, slot_size)

    out = decoder(z)

    C, H, W = out.shape[1:]

    out = out.view((bs, num_slots, C, H, W))
    assert(len(out.shape) == 5)

    masks = out[:, :, -1:, :, :]
    masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]

    channel_idx = 0

    recons_rgb = out[:, :, channel_idx:channel_idx + 3, :, :]
    recon_rgb_combined = torch.sum(recons_rgb * masks, dim=1)
    channel_idx += 3

    res = {'post_rgb_recon_combined': recon_rgb_combined, 'post_rgb_recons': recons_rgb}

    return res

class VPBaseModel(pl.LightningModule):
    def __init__(
            self, optimizer=None,
            train_metrics=None, val_metrics=None, additional_metrics_last_val=None,
            total_steps=None, num_val_frames=None, max_instances=None,
            **kwargs):
        super().__init__()

        self.optimizer_config = optimizer
        self.total_steps = total_steps
        self.num_val_frames = num_val_frames
        self.final_validation = False
        self.val_metrics_config = val_metrics
        self.metrics_last_val_config = additional_metrics_last_val
        self.max_instances = max_instances

        if self.metrics_last_val_config is not None:
            self.metrics_last_val_config.update(val_metrics)

        self.train_metrics = None
        self.val_metrics = None
        self.setup_metrics(train_metrics=train_metrics, val_metrics=val_metrics)

        self.last_val_metrics = None
        self.last_val_metrics_framewise = None

        self.save_hyperparameters(logger=False)

    def setup_metrics(self, train_metrics, val_metrics):
        self.train_metrics = torch.nn.ModuleDict({k: v(max_instances=self.max_instances) for k,v in train_metrics.items()}) if train_metrics is not None else None
        self.val_metrics = torch.nn.ModuleDict({k: v(max_instances=self.max_instances) for k,v in val_metrics.items()}) if val_metrics is not None else None

    def get_loss(self, batch, outputs, dataloader_idx):
        loss_dict = {}
        pred = outputs['post_rgb_recon_combined']
        target = batch['unroll']['video'] if 'burn_in' in batch else batch['video']
        loss_dict['post_rgb_recon'] = F.mse_loss(pred, target)

        pred = outputs['burn_in']['post_rgb_recon_combined']
        target = batch['burn_in']['video']
        loss_dict['burn_in_post_rgb_recon'] = F.mse_loss(pred, target)

        return loss_dict

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        out_dict = self(batch)

        loss_dict = self.get_loss(batch, out_dict, dataloader_idx=dataloader_idx)
        loss = 0
        for v in loss_dict.values():
            loss += v

        batch = batch['unroll']

        outputs = {'loss': loss,
                   'batch_idx': batch_idx, 'loss_dict': loss_dict}

        outputs['video'] = batch['video']

        outputs.update(out_dict)
        self.log_training(outputs)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out_dict = self(batch, dataloader_idx=dataloader_idx)

        loss_dict = self.get_loss(batch, out_dict, dataloader_idx=dataloader_idx)
        loss = 0
        for v in loss_dict.values():
            loss += v

        batch = batch['unroll']

        outputs = {'loss': loss,
                   'batch_idx': batch_idx, 'loss_dict': loss_dict}

        outputs['video'] = batch['video']

        outputs.update(out_dict)
        self.log_validation(outputs, dataloader_idx=dataloader_idx)

        return outputs

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint['state_dict'].keys())
        for k in keys:
            if k.startswith('train_metrics') or k.startswith('val_metrics') or k.startswith('last_val_metrics'):
                del checkpoint['state_dict'][k]

    def get_params_for_optimizer(self):
        return [{'params': list(filter(lambda p: p.requires_grad,
                                     self.parameters()))}]


    def configure_optimizers(self):
        assert(self.optimizer_config is not None)

        params = self.get_params_for_optimizer()

        optimizer = optim.Adam(params=params, lr=self.optimizer_config['lr'])

        result = {'optimizer': optimizer}

        return result

    def setup_final_validation(self):
        assert(self.num_val_frames is not None)
        self.final_validation = True
        # Setup "best" metrics
        print("Preparing last validation with the following metrics:")
        print(self.metrics_last_val_config.keys() if self.metrics_last_val_config is not None else None)
        self.last_val_metrics = torch.nn.ModuleDict({
            k: v(max_instances=self.max_instances).to(self.device) for k,v in self.metrics_last_val_config.items()}) \
            if self.metrics_last_val_config is not None else None
        self.last_val_metrics_framewise = torch.nn.ModuleDict({
            k: torch.nn.ModuleList([v(max_instances=self.max_instances).to(self.device) for _ in range(
                self.num_val_frames)]) for k, v in self.metrics_last_val_config.items()}) \
                if self.metrics_last_val_config is not None else None


    def log_training(self, outputs):
        self.log('train/loss_step', outputs['loss'], on_epoch=False, on_step=True,
                     prog_bar=True, sync_dist=False, add_dataloader_idx=False)
        self.log('train/loss_epoch', outputs['loss'], on_step=False,
                 on_epoch=True,
                 prog_bar=True, sync_dist=True, add_dataloader_idx=False)

        for k, v in outputs['loss_dict'].items():
            self.log(f'train/loss/{k}', v, on_epoch=True, on_step=False,
                     sync_dist=True, add_dataloader_idx=False)

        if self.trainer.is_global_zero:
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                     rank_zero_only=True, add_dataloader_idx=False)

        for metric_name in self.train_metrics.keys():
            self.train_metrics[metric_name](**outputs)
            self.log(f'train/{metric_name}', self.train_metrics[metric_name],
                     on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

    def log_validation(self, outputs, dataloader_idx=0):
        if self.final_validation:
            assert(self.last_val_metrics is not None)
            assert(self.last_val_metrics_framewise is not None)

            for metric_name in self.last_val_metrics.keys():
                if '/' in metric_name and dataloader_idx == 0:
                    continue

                self.last_val_metrics[metric_name](**outputs)
                self.log(f'val/best/{metric_name}', self.last_val_metrics[metric_name],
                         on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

            self.log('val/best/loss', outputs['loss'], on_epoch=True, on_step=False,
                     prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            for k, v in outputs['loss_dict'].items():
                self.log(f'val/best/loss/{k}', v, on_epoch=True, on_step=False,
                         sync_dist=True, add_dataloader_idx=False)

            for metric_name in self.last_val_metrics_framewise.keys():
                for i in range(len(self.last_val_metrics_framewise[metric_name])):
                    self.last_val_metrics_framewise[metric_name][i](**get_output_slice(outputs, i))
                    self.log(f'val/best/{metric_name}/{i+1}', self.last_val_metrics_framewise[metric_name][i], add_dataloader_idx=False,
                             on_step=False, on_epoch=True, sync_dist=True)
        else:
            self.log('val/loss', outputs['loss'], on_epoch=True, on_step=False,
                         prog_bar=True, sync_dist=True, add_dataloader_idx=False)
            for k, v in outputs['loss_dict'].items():
                self.log(f'val/loss/{k}', v, on_epoch=True, on_step=False,
                         sync_dist=True, add_dataloader_idx=False)

            for metric_name in self.val_metrics.keys():
                self.val_metrics[metric_name](**outputs)
                self.log(f'val/{metric_name}', self.val_metrics[metric_name],
                         on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)


class BaseVIP(VPBaseModel):
    def __init__(
            self,
            encoder,
            decoder,
            predictor,
            num_context_frames,
            max_concurrent_frames=12, # During validation due to memory concerns
            **kwargs
            ):
        super().__init__(**kwargs)

        self.max_concurrent_frames = max_concurrent_frames
        self.num_context_frames = num_context_frames

        assert(self.max_concurrent_frames >= self.num_context_frames)

        self.predictor = init_module(predictor)
        self.encoder = init_module(encoder)
        self.decoder = init_module(decoder)

    def encode(self, img, batch, dataloader_idx, prev_state=None):
        assert(len(img.shape) == 5)
        T = img.shape[1]
        if T <= self.max_concurrent_frames or self.training:
            return self._encode(img, batch=batch, prev_state=prev_state,
                                dataloader_idx=dataloader_idx, frame_start_idx=0)

        cat_dict = None
        for clip_idx in range(0, T, self.max_concurrent_frames):
            output = self._encode(img[:, clip_idx:clip_idx + self.max_concurrent_frames],
                                      prev_state=prev_state, batch=batch,
                                    dataloader_idx=dataloader_idx, frame_start_idx=clip_idx)

            prev_state = output
            output = detach_dict(output)

            if cat_dict is None:
                cat_dict = {k: [v] if v is not None else v for k, v in output.items()}
            else:
                for k, v in output.items():
                    if v is not None:
                        cat_dict[k].append(v)

            torch.cuda.empty_cache()

        assert(cat_dict is not None)
        cat_dict = concat_dict(cat_dict, dim=1)

        return cat_dict

    def rollout(self, pred_len, batch, frame_start_idx, previous_state, dataloader_idx=0):
        result = self._rollout(previous_state=previous_state,
                                pred_len=pred_len, batch=batch, dataloader_idx=dataloader_idx,
                                frame_start_idx=frame_start_idx)
        pred_z = result.pop('pred_z')
        B = batch['unroll']['video'].shape[0]
        T = pred_len

        out_dict = self.decode(pred_z.flatten(0, 1))

        out_dict = {k: v.unflatten(0, (B, T)) if v is not None else v for k, v in out_dict.items()}
        out_dict['pred_z'] = pred_z
        out_dict.update(result)

        return out_dict

    def forward(self, data_dict, dataloader_idx=0):
        # Video Prediction Mode
        B, T_unroll = data_dict['unroll']['video'].shape[:2]
        T_burn_in = data_dict['burn_in']['video'].shape[1]

        burn_in_video = data_dict['burn_in']['video']

        # Encode burn in frames
        burn_in_encoded = self.encode(
            img=burn_in_video,
            batch=data_dict['burn_in'],
            dataloader_idx=dataloader_idx,
            prev_state=None)

        burn_in_decoded = self.decode(burn_in_encoded['pred_z'].flatten(0, 1))

        previous_state = burn_in_encoded
        if T_unroll <= self.max_concurrent_frames or self.training:
            out_dict = self.rollout(
                previous_state=previous_state, pred_len=T_unroll, batch=data_dict,
                dataloader_idx=dataloader_idx, frame_start_idx=0)
        else:
            # Split along temporal dim
            cat_dict = None
            for clip_idx in range(0, T_unroll, self.max_concurrent_frames):
                output = self.rollout(
                    pred_len=min(self.max_concurrent_frames, T_unroll - clip_idx), batch=data_dict,
                    dataloader_idx=dataloader_idx, frame_start_idx=clip_idx,
                    previous_state=previous_state)

                previous_state = output

                # because this should be in test mode, we detach the outputs
                output = detach_dict(output)
                if cat_dict is None:
                    cat_dict = {k: [v] if v is not None else v for k, v in output.items()}
                else:
                    for k, v in output.items():
                        if v is not None:
                            cat_dict[k].append(v)

                torch.cuda.empty_cache()
            assert(cat_dict is not None)
            out_dict = concat_dict(cat_dict, dim=1)

        burn_in_encoded.update(
            {k: v.unflatten(0, (B, T_burn_in)) if v is not None else v
             for k, v in burn_in_decoded.items()})

        burn_in_encoded.pop('pred_z')
        out_dict['burn_in'] = burn_in_encoded
        torch.cuda.empty_cache()

        return out_dict

    def _forward(self, batch, dataloader_idx, frame_idx_start, prev_state=None):
        img = batch['video'][:, frame_idx_start:frame_idx_start + self.max_concurrent_frames]
        assert(len(img.shape) == 5)

        B, T = img.shape[:2]
        out_dict = \
            self.encode(img, prev_state=prev_state, batch=batch,
                        dataloader_idx=dataloader_idx)

        post_dict = self.decode(out_dict['pred_z'].flatten(0, 1))

        out_dict.update(
            {k: v.unflatten(0, (B, T)) if v is not None else v
             for k, v in post_dict.items()})

        return out_dict

    def decode(self, z):
        assert(len(z.shape) == 2)
        recons_rgb = self.decoder(z)

        recon_rgb_combined = recons_rgb

        res = {'post_rgb_recon_combined': recon_rgb_combined, 'post_rgb_recons': recons_rgb, }

        return res

    def _encode(self, img, batch, dataloader_idx, frame_start_idx, prev_state=None):
        assert(len(img.shape) == 5)
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        z = self.encoder(img)
        z = z.unflatten(0, (B, T))

        return {'pred_z': z}

    def _rollout(self, pred_len, batch, previous_state, frame_start_idx, dataloader_idx=0):
        z = previous_state['pred_z'][:, -self.num_context_frames:]

        # generate future z autoregressively
        pred_out = []
        for _ in range(pred_len):
            assert(self.predictor is not None)
            z_next = self.predictor(z)[:, -1]  # [B, N, C]
            assert(len(z_next.shape) == 2)

            pred_out.append(z_next)

            # feed the predicted slots autoregressively
            z = torch.cat([z[:, 1:], pred_out[-1].unsqueeze(1)], dim=1)

        return {'pred_z': torch.stack(pred_out, dim=1)}

class SlotVIP(BaseVIP):
    def __init__(
            self,
            slot_size,
            num_slots,
            slot_attention,
            **kwargs
            ):
        self.slot_size = slot_size
        self.num_slots = num_slots

        super().__init__(**kwargs, max_instances=self.num_slots - 1)

        self.slot_attention = init_module(slot_attention, in_features=self.slot_size,
                                          num_slots=self.num_slots, slot_size=self.slot_size)

        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)), requires_grad=True)

    def _encode(self, img, batch, dataloader_idx, frame_start_idx, prev_state=None):
        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        encoder_out = self.encoder(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # init slots
        if prev_state is None:
            prev_z = None

            assert(self.init_latents is not None)
            init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]
        else:
            init_latents = None
            prev_z = prev_state['pred_z'][:, -self.num_context_frames:]

        # apply SlotAttn on video frames via reusing slots
        all_pred_z = []
        for idx in range(T):
            # init
            if prev_z is None:
                assert(init_latents is not None)
                latents = init_latents  # [B, N, C]
            else:
                # Use up to self.num_burn_in_frames of representations as reference
                x = prev_z[:, -self.num_context_frames:]

                assert(self.predictor is not None)
                latents = self.predictor(x)[:, -1]  # [B, N, C]
                assert(len(latents.shape) == 3)

            # (B, T, slot_size)
            pred_z = self.slot_attention(encoder_out[:, idx], latents)
            all_pred_z.append(pred_z)

            # next timestep
            prev_z = torch.cat([prev_z, pred_z.unsqueeze(1)], dim=1) if prev_z is not None else pred_z.unsqueeze(1)

        # (B, T, self.num_slots, self.slot_size)
#        kernel_dist = torch.stack(all_kernel_dist, dim=1)
        pred_z = torch.stack(all_pred_z, dim=1)

        return {'pred_z': pred_z}

    def decode(self, z):
        return slot_decode(z=z, decoder=self.decoder)

class ProcVIP(BaseVIP):
    def __init__(
            self,
            state_autoencoder_alignment_factor=0.0,
            learned_parameter_specific_lr=None,
            **kwargs
            ):
        super().__init__(**kwargs)

        self.state_autoencoder_alignment_factor = state_autoencoder_alignment_factor
        self.learned_parameter_specific_lr = learned_parameter_specific_lr

        def fn(a, b):
            err = torch.pow(a - b, 2)
            return err.mean()
        self.state_loss_fn = fn

    def log_training(self, outputs):
        super().log_training(outputs)

        if hasattr(self.predictor.Fs[0], 'learned_parameters') and self.predictor.Fs[0].learned_parameters:
            for k, v in self.predictor.Fs[0].learned_parameters.items():
                self.log(f'learned_params/{k}', elup1(v), on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

    def get_params_for_optimizer(self):
        if self.learned_parameter_specific_lr is None:
            params = [{'params': list(filter(lambda p: p.requires_grad,
                                     self.parameters()))}]
        else:
            params_list = filter(lambda p: p[1].requires_grad and not p[0].startswith('predictor.Fs.0.learned_parameters.'),
                                     self.named_parameters())
            params_list = [p[1] for p in params_list]
            F_params = filter(lambda p: p[1].requires_grad and p[0].startswith('predictor.Fs.0.learned_parameters.'),
                                     self.named_parameters())
            F_params = [p[1] for p in F_params]

            assert(F_params)

            params = [{'params': params_list}, {'params': F_params, 'lr': self.learned_parameter_specific_lr['lr']}]

        return params

    def _rollout_iteration(self, in_x, sym_state,
                           dataloader_idx, frame_idx):
        z_out, sym_state, decoded_state = self.predictor(
            z=in_x, z_a_sym_dict=sym_state,
            dataloader_idx=dataloader_idx, frame_idx=frame_idx)
        # feed the predicted slots autoregressively
        in_x = torch.cat([in_x[:, 1:], z_out], dim=1)

        return in_x, sym_state, decoded_state

    def get_loss(self, batch, outputs, dataloader_idx):
        loss_dict = super().get_loss(batch, outputs, dataloader_idx)

        if self.state_autoencoder_alignment_factor:
            # predicted state vs. decoded state
            decoded_sym_states = torch.cat([
                outputs['decoded_sym_states'][k] for k in self.predictor.Fs[0].keys], dim=-1)
            predicted_sym_states = torch.cat([
                outputs['sym_states'][k] for k in self.predictor.Fs[0].keys], dim=-1)

            decoded_sym_states = torch.cat([
                torch.cat([outputs['burn_in']['decoded_sym_states'][k] for k in self.predictor.Fs[0].keys],
                          dim=-1), decoded_sym_states], dim=1)
            predicted_sym_states = torch.cat([
                torch.cat([outputs['burn_in']['sym_states'][k] for k in self.predictor.Fs[0].keys],
                          dim=-1), predicted_sym_states], dim=1)

            state_autoencoder_alignment_loss = self.state_loss_fn(decoded_sym_states, predicted_sym_states)

            state_autoencoder_alignment_loss *= self.state_autoencoder_alignment_factor

            loss_dict['state_autoencoder_alignment'] = state_autoencoder_alignment_loss

        return loss_dict

    def _rollout(self, pred_len, batch, previous_state, frame_start_idx, dataloader_idx=0):
        in_x = previous_state['z_encoded'][:, -self.num_context_frames:]

        sym_state = {k: v[:, -1] for k, v in previous_state['sym_states'].items()}

        # generate future slots autoregressively
        pred_out = []
        sym_states = []
        decoded_sym_states = []

        for i in range(frame_start_idx, frame_start_idx + pred_len):
            in_x, sym_state, decoded_state = self._rollout_iteration(
                in_x=in_x, sym_state=sym_state,
                dataloader_idx=dataloader_idx, frame_idx=i)

            pred_out.append(in_x[:, -1])

            decoded_sym_states.append(decoded_state)
            sym_states.append(sym_state)

        res = {'pred_z': self.predictor.P_out(torch.stack(pred_out, dim=1)),
               'z_encoded': torch.stack(pred_out, dim=1),
               'sym_states': {k: torch.stack([v[k] for v in sym_states], dim=1) for k in sym_states[0].keys()},
               'groundtruth_sym_states': {k: batch['unroll'][k][:, frame_start_idx:frame_start_idx + pred_len] for k in sym_states[0].keys()},}

        if decoded_sym_states:
            res['decoded_sym_states'] = {k: torch.stack([v[k] for v in decoded_sym_states], dim=1) for k in decoded_sym_states[0].keys()}

        return res

    def _encode(self, img, batch, dataloader_idx, frame_start_idx, prev_state=None):
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        z = self.encoder(img)
        z = z.unflatten(0, (B, T))
        z_encoded = self.predictor.P_in(z)
        # `encoder_out` has shape: [B, T, H*W, out_features]

        sym_states = []
        decoded_sym_states = []

        # init
        prev_z = None
        if prev_state is None:
            conditioning = self.predictor.Fs[0].get_state_tensor(
                batch=batch, prepend_background=False, keep_object_dim=False)[:, 0]

            sym_state = self.predictor.Fs[0].convert_tensor_to_state_dict(
                conditioning, has_object_dim=False)

            if self.state_autoencoder_alignment_factor:
                decoded_sym_states.append(self.predictor.read_sym_state(z_encoded[:, 0]))
        else:
            prev_z = prev_state['z_encoded'][:, -self.num_context_frames:]

            sym_state = {k: v[:,-1] for k, v in prev_state['sym_states'].items()}

        for idx in range(T):
            if prev_z is None:
                # First iteration; skip prediction
                sym_states.append(sym_state)

                prev_z = z_encoded[:, :1]

                continue

            # We just need to obtain current sym state
            _, sym_state, decoded_state = self._rollout_iteration(
                in_x=prev_z[:, -self.num_context_frames:], sym_state=sym_state,
                dataloader_idx=dataloader_idx,
                frame_idx=frame_start_idx + idx)

            decoded_sym_states.append(decoded_state)

            sym_states.append(sym_state)

            # next timestep
            prev_z = torch.cat([prev_z, z_encoded[:, idx:idx+1]], dim=1)

        res = {
            'pred_z': z,
            'z_encoded': z_encoded,
            'sym_states': {k: torch.stack([v[k] for v in sym_states], dim=1) for k in sym_states[0].keys()},
            'groundtruth_sym_states': {k: batch[k][:, frame_start_idx:frame_start_idx + T] for k in sym_states[0].keys()},
        }

        if decoded_sym_states:
            res['decoded_sym_states'] = {k: torch.stack([v[k] for v in decoded_sym_states], dim=1) for k in decoded_sym_states[0].keys()}

        return res

class ProcSlotVIP(ProcVIP):
    def __init__(
            self,
            slot_size,
            num_slots,
            slot_attention,
            **kwargs
            ):
        self.slot_size = slot_size
        self.num_slots = num_slots

        super().__init__(**kwargs, max_instances=self.num_slots - 1)

        self.slot_attention = init_module(slot_attention, in_features=self.slot_size,
                                          num_slots=self.num_slots, slot_size=self.slot_size)

        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)), requires_grad=True)

    def setup_metrics(self, train_metrics, val_metrics):
        self.train_metrics = torch.nn.ModuleDict({k: v(max_instances=self.num_slots-1) for k,v in train_metrics.items()}) if train_metrics is not None else None
        self.val_metrics = torch.nn.ModuleDict({k: v(max_instances=self.num_slots-1) for k,v in val_metrics.items()}) if val_metrics is not None else None

    def _encode(self, img, batch, dataloader_idx, frame_start_idx, prev_state=None):
        B, T, C, H, W = img.shape

        img = img.flatten(0, 1)

        z = self.encoder(img)
        z = z.unflatten(0, (B, T))

        all_pred_z = []
        sym_states = []
        decoded_sym_states = []
        all_encoded_z = []

        # init slots
        if prev_state is None:
            prev_z = None

            conditioning = self.predictor.Fs[0].get_state_tensor(batch=batch, prepend_background=True, keep_object_dim=True)[:, 0]
            init_latents = self.init_latents.repeat(conditioning.shape[0], 1, 1)

            sym_state = self.predictor.Fs[0].convert_tensor_to_state_dict(
                conditioning, has_object_dim=True)

            if self.state_autoencoder_alignment_factor:
                decoded_sym_states.append(self.predictor.read_sym_state(init_latents))
        else:
            init_latents = None
            prev_z = self.predictor.P_in(prev_state['pred_z'][:, -self.num_context_frames:])

            sym_state = {k: v[:,-1] for k, v in prev_state['sym_states'].items()}

        latents = None
        for idx in range(T):
            # init
            if prev_z is None:
                assert(init_latents is not None)
                latents = init_latents  # [B, N, C]

            else:
                # Use up to self.num_burn_in_frames of representations as reference
                x = prev_z[:, -self.num_context_frames:]

                latents, sym_state, decoded_state = self._rollout_iteration(
                    in_x=x, sym_state=sym_state,
                    dataloader_idx=dataloader_idx,
                    frame_idx=frame_start_idx + idx)

                decoded_sym_states.append(decoded_state)

                latents = latents[:, -1]

            all_encoded_z.append(latents)

            pred_z = self.slot_attention(z[:, idx], self.predictor.P_out(
                latents))

            encoded_pred_z = self.predictor.P_in(pred_z)

            sym_states.append(sym_state)

            all_pred_z.append(pred_z)

            # next timestep
            prev_z = torch.cat([prev_z, encoded_pred_z.unsqueeze(1)], dim=1) if prev_z is not None else encoded_pred_z.unsqueeze(1)

        pred_z = torch.stack(all_pred_z, dim=1)

        res = {
            'pred_z': pred_z,
            'z_encoded': torch.stack(all_encoded_z, dim=1),
            'sym_states': {k: torch.stack([v[k] for v in sym_states], dim=1) for k in sym_states[0].keys()},
            'groundtruth_sym_states': {k: batch[k][:, frame_start_idx:frame_start_idx + T] for k in sym_states[0].keys()},
        }

        if decoded_sym_states:
            res['decoded_sym_states'] = {k: torch.stack([v[k] for v in decoded_sym_states], dim=1) for k in decoded_sym_states[0].keys()}

        return res

    def decode(self, z):
        return slot_decode(z=z, decoder=self.decoder)
