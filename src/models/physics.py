import pytorch_lightning as pl

import torch

import torch
from torch import nn
from src.lib import elup1

class PhysicsEngine(pl.LightningModule):
    def __init__(self, G, step_rate, frame_rate, obj_mass,
                 learned_parameters=None,
                 prepend_background=True):
        super().__init__()

        self.keys = ['positions', 'velocities']
        self.prepend_background = prepend_background
        self.has_object_dim = True

        if learned_parameters is None:
            self.learned_parameters = None
        else:
            self.learned_parameters = nn.ParameterDict()
            for p in learned_parameters:
                self.learned_parameters[p] = nn.Parameter(nn.init.normal_(torch.empty(1)))

        self.config = {
            'G': G,
            'step_rate': step_rate,
            'frame_rate': frame_rate,
            'simulation_steps': step_rate // frame_rate,
            'frame_rate_dt': 1.0 / frame_rate,
            'obj_mass': obj_mass}
        assert(self.config['simulation_steps'] >= 1)
        self.config['simulation_dt'] = 1.0 / float(step_rate)

    def convert_state_dict_to_tensor(self, d, keep_object_dim):
        res = torch.cat([d[k] for k in self.keys], dim=-1)

        if not keep_object_dim:
            res = torch.flatten(res, 1)

        return res

    def convert_tensor_to_state_dict(self, z, has_object_dim):
        if not has_object_dim:
            z = z.view(z.shape[:-1] + (-1, 6))
        assert(z.shape[-1] == 6)

        return {'positions': z[..., :3], 'velocities': z[..., 3:]}

    def get_sym_state_size(self):
        return 2 * 3

    def get_state_tensor(self, batch, prepend_background, keep_object_dim):
        assert(prepend_background == self.prepend_background)
        res = torch.cat([batch['positions'], batch['velocities']], dim=-1)
        if prepend_background:
            res = torch.cat([torch.zeros_like(res[:, :, 0:1]), res], dim=2)

        if not keep_object_dim:
            res = torch.flatten(res, 2)

        return res

    def forward(self, previous_states, frame_idx):
        # Presence filter with interval [-1, 1] describes object certainty
        vel = previous_states['velocities']
        pos = previous_states['positions']

        if self.learned_parameters:
            if 'G' in self.learned_parameters:
                G = elup1(self.learned_parameters['G'])
            else:
                G = self.config['G']

            if 'obj_mass' in self.learned_parameters:
                obj_mass = elup1(self.learned_parameters['obj_mass'])
            else:
                obj_mass = self.config['obj_mass']
        else:
            G = self.config['G']
            obj_mass = self.config['obj_mass']

        simulation_dt = self.config['simulation_dt']
        simulation_steps = self.config['simulation_steps']

        mass = torch.full_like(pos[...,:1], fill_value=obj_mass, device=self.device)
        if self.prepend_background:
            mass[:,0] = 0.0

        ext_pos = torch.cat([torch.zeros_like(pos[:,0:1], device=self.device, dtype=pos.dtype), pos], dim=-2)
        # We have 3D positions, but only consider x and y for force calculation since z does not change for the objects in order for them to better stay in view
        ext_pos = torch.cat([ext_pos[...,:-1], torch.zeros_like(ext_pos[...,-1:])], dim=-1)

        # Add ghost mass
        ext_mass = torch.cat([torch.full_like(mass[:,0:1], fill_value=2.0, device=self.device, dtype=mass.dtype), mass], dim=-2)

        for sim_idx in range(simulation_steps):
            # (1, 9, 9, 3)
            p_diffs = torch.unsqueeze(ext_pos, dim=-3) - torch.unsqueeze(ext_pos, dim=-2)
            # (1, 9, 9, 1)
            r2 = torch.sum(torch.pow(p_diffs, 2), dim=-1, keepdim=True) + 1

            # (1, 9, 9, 1)
            norm = torch.sqrt(r2)

            # (1, 9, 9, 3)
            F_dir = p_diffs / norm

            # (1, 9, 9, 1)
            m_prod = torch.unsqueeze(ext_mass, dim=-3) * torch.unsqueeze(ext_mass, dim=-2)

            F = torch.sum(torch.where(m_prod > 0, F_dir * (G * (m_prod / (r2))), 0.0), dim=-2)[...,1:,:]

            # F = ma
            a = F / (mass + 1e-9)

            # Semi implicit euler
            vel = torch.where(mass > 0, vel + simulation_dt * a, vel)
            pos = torch.where(mass > 0, pos + simulation_dt * vel, pos)

            ext_pos = torch.cat([torch.zeros_like(pos[:, 0:1], device=self.device, dtype=pos.dtype), pos], dim=-2)
            ext_pos = torch.cat([ext_pos[...,:-1], torch.zeros_like(ext_pos[...,-1:])], dim=-1)

        res = {'positions': pos, 'velocities': vel}

        return res

class AcrobotPhysicsEngine(pl.LightningModule):
    def __init__(self, step_rate, frame_rate,
                 link_len,
                 prepend_background=True):
        super().__init__()

        self.has_object_dim = True

        self.config = {
            'link_len': link_len,
            'step_rate': step_rate,
            'frame_rate': frame_rate,
            'simulation_steps': step_rate // frame_rate,
            'frame_rate_dt': 1.0 / frame_rate,
            'obj_mass': 1.5
        }

        self.prepend_background = prepend_background

        assert(self.config['simulation_steps'] >= 1)
        self.config['simulation_dt'] = 1.0 / float(step_rate)

        self.keys = ['theta', 'dtheta']

    def convert_state_dict_to_tensor(self, d, keep_object_dim):
        res = torch.cat([d[k] for k in self.keys], dim=-1)

        if not keep_object_dim:
            res = torch.flatten(res, 1)

        return res

    def convert_tensor_to_state_dict(self, z, has_object_dim):
        if not has_object_dim:
            z = z.view(z.shape[:-1] + (-1, 2))
        assert(z.shape[-1] == 2)
        assert(z.shape[-2] == (3 if self.prepend_background else 2))

        return {'theta': z[..., :1], 'dtheta': z[..., 1:]}

    def get_sym_state_size(self):
        return 1 + 1

    def get_state_tensor(self, batch, prepend_background, keep_object_dim):
        assert(prepend_background == self.prepend_background)

        res = torch.cat([batch['theta'], batch['dtheta']], dim=-1)
        if prepend_background:
            res = torch.cat([torch.zeros_like(res[:, :, 0:1]), res], dim=2)

        if not keep_object_dim:
            res = torch.flatten(res, 2)

        return res


    def forward(self, previous_states, frame_idx, **kwargs):
        if self.prepend_background:
            _, theta1, theta2 = torch.split(previous_states['theta'], 1, dim=1)
            _, dtheta1, dtheta2 = torch.split(previous_states['dtheta'], 1, dim=1)
        else:
            theta1, theta2 = torch.split(previous_states['theta'], 1, dim=1)
            dtheta1, dtheta2 = torch.split(previous_states['dtheta'], 1, dim=1)

        theta1 = torch.squeeze(theta1, dim=1)
        theta2 = torch.squeeze(theta2, dim=1)
        dtheta1 = torch.squeeze(dtheta1, dim=1)
        dtheta2 = torch.squeeze(dtheta2, dim=1)

        obj_mass = self.config['obj_mass']
        link_len = self.config['link_len']
        lc = 0.5
        I = 1.0
        g = 9.8

        simulation_dt = self.config['simulation_dt']
        lc1 = lc
        lc2 = lc
        I1 = I
        I2 = I
        l1 = link_len
        l2 = link_len
        m1 = obj_mass
        m2 = obj_mass

        simulation_steps = self.config['simulation_steps']

        for i in range(simulation_steps):
            d1 = (
                m1 * lc1**2
                + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * torch.cos(theta2))
                + I1
                + I2
                )
            d2 = m2 * (lc2**2 + l1 * lc2 * torch.cos(theta2)) + I2
            phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - torch.pi / 2.0)
            phi1 = (
                -m2 * l1 * lc2 * dtheta2**2 * torch.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - torch.pi / 2)
                + phi2
            )
            ddtheta2 = (
                    d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * torch.sin(theta2) - phi2
                ) / (m2 * lc2**2 + I2 - d2**2 / d1)
            ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

            dtheta1 = dtheta1 + ddtheta1 * simulation_dt
            dtheta2 = dtheta2 + ddtheta2 * simulation_dt

            new_theta1 = theta1 + dtheta1 * simulation_dt
            new_theta2 = theta2 + dtheta2 * simulation_dt

            # Clip / Wrap values
            new_theta1 = wrap(new_theta1)
            new_theta2 = wrap(new_theta2)
            dtheta1 = torch.clip(dtheta1, -4.0 * torch.pi, 4.0 * torch.pi)
            dtheta2 = torch.clip(dtheta2, -9.0 * torch.pi, 9.0 * torch.pi)

            theta1 = new_theta1
            theta2 = new_theta2


        res = {'theta': torch.stack([previous_states['theta'][:,0], theta1, theta2] if self.prepend_background else [theta1, theta2], dim=1), 'dtheta': torch.stack([previous_states['dtheta'][:,0], dtheta1, dtheta2] if self.prepend_background else [dtheta1, dtheta2], dim=1)}

        return res

class CameraPendulum(pl.LightningModule):
    def __init__(self, step_rate, frame_rate,
                 link_len,
                 ):
        super().__init__()

        self.has_object_dim = False

        self.config = {
            'link_len': link_len,
            'step_rate': step_rate,
            'frame_rate': frame_rate,
            'simulation_steps': step_rate // frame_rate,
            'frame_rate_dt': 1.0 / frame_rate,
            'obj_mass': 1.5
        }
        assert(self.config['simulation_steps'] >= 1)
        self.config['simulation_dt'] = 1.0 / float(step_rate)

        self.keys = ['camera_positions', 'camera_thetas', 'camera_dthetas']

    def convert_state_dict_to_tensor(self, d, keep_object_dim):
        assert(keep_object_dim == False)
        res = torch.cat([d['camera_positions'], d['camera_thetas'], d['camera_dthetas']], dim=-1)
        assert(res.shape[-1] == self.get_sym_state_size())

        return res

    def convert_tensor_to_state_dict(self, z, has_object_dim):
        assert(has_object_dim == False)
        assert(z.shape[-1] == self.get_sym_state_size())

        thetas = z[..., 3:5]
        dthetas = z[..., 5:]

        return {'camera_positions': z[..., 0:3],
                'camera_thetas': thetas,
                'camera_dthetas': dthetas}

    def get_sym_state_size(self):
        return 7

    def get_state_tensor(self, batch, prepend_background, keep_object_dim):
        assert(prepend_background == False)
        assert(keep_object_dim == False)

        res = torch.cat([batch['camera_positions'], batch['camera_thetas'], batch['camera_dthetas']], dim=-1)

        assert(res.shape[-1] == self.get_sym_state_size())
        assert(len(res.shape) == 3)

        return res

    def forward(self, previous_states, frame_idx, **kwargs):
        theta1, theta2 = torch.split(previous_states['camera_thetas'], 1, dim=-1)
        dtheta1, dtheta2 = torch.split(previous_states['camera_dthetas'], 1, dim=-1)

        obj_mass = 1.5
        link_len = self.config['link_len']
        lc = 0.5
        I = 1.0
        g = 9.8

        simulation_dt = self.config['simulation_dt']
        lc1 = lc
        lc2 = lc
        I1 = I
        I2 = I
        l1 = link_len
        l2 = link_len
        m1 = obj_mass
        m2 = obj_mass

        simulation_steps = self.config['simulation_steps']

        for i in range(simulation_steps):
            d1 = (
                m1 * lc1**2
                + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * torch.cos(theta2))
                + I1
                + I2
                )
            d2 = m2 * (lc2**2 + l1 * lc2 * torch.cos(theta2)) + I2
            phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - torch.pi / 2.0)
            phi1 = (
                -m2 * l1 * lc2 * dtheta2**2 * torch.sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - torch.pi / 2)
                + phi2
            )
            ddtheta2 = (
                    d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * torch.sin(theta2) - phi2
                ) / (m2 * lc2**2 + I2 - d2**2 / d1)
            ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

            dtheta1 = dtheta1 + ddtheta1 * simulation_dt
            dtheta2 = dtheta2 + ddtheta2 * simulation_dt

            new_theta1 = theta1 + dtheta1 * simulation_dt
            new_theta2 = theta2 + dtheta2 * simulation_dt

            # Clip / Wrap values
            new_theta1 = wrap(new_theta1)
            new_theta2 = wrap(new_theta2)
            dtheta1 = torch.clip(dtheta1, -4.0 * torch.pi, 4.0 * torch.pi)
            dtheta2 = torch.clip(dtheta2, -9.0 * torch.pi, 9.0 * torch.pi)

            theta1 = new_theta1
            theta2 = new_theta2

        # Camera position
        camera_pos = torch.cat([
                -2.*link_len * torch.sin(theta1) - link_len * torch.sin(theta1 + theta2),
                 2.*link_len * torch.cos(theta1) + link_len * torch.cos(theta1 + theta2),
                torch.full(theta1.shape, 10.0, device=self.device, dtype=torch.float32)], dim=-1)


        res = {'camera_thetas': torch.cat([theta1, theta2], dim=-1), 'camera_dthetas': torch.cat([dtheta1, dtheta2], dim=-1),
               'camera_positions': camera_pos}

        return res

def wrap(x):
    return torch.arctan2(torch.sin(x), torch.cos(x))
