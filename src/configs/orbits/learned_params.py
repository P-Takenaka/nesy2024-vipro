import ml_collections
import os
import functools
import pickle

from src.lib import LPIPS, SSIM, PSNR

def get_config():
  config = ml_collections.ConfigDict()

  config.experiment_name = 'orbits'
  config.num_steps = 500000
  config.max_grad_norm = 0.05
  config.log_n_samples = 5
  config.early_stopping = True

  resolution = (64, 64)

  num_burn_in_frames = 6
  num_train_frames = 12
  num_val_frames = 24

  z_a_size = 64
  z_b_size = 64
  z_c_size = 64

  slot_size = z_a_size + z_b_size + z_c_size

  config.data = ml_collections.ConfigDict({
        "data_dir": os.environ["DATA_DIR"],
        "name": "orbits",
        "batch_size": 32, # Total
        "num_train_frames": num_train_frames,
        "num_val_frames": num_val_frames,
        "num_burn_in_frames": num_burn_in_frames,
        "load_keys": ["video", "physics"],
        "target_size": resolution,
  })

  with open(os.path.join(config.data.data_dir, config.data.name, '0', 'metadata.pkl'), 'rb') as f:
      sample_metadata = pickle.load(f)

  config.model = ml_collections.ConfigDict({
        "module": "src.models.ProcSlotVIP",
        "slot_size": slot_size,
        "do_segmentation": False,
        "num_context_frames": num_burn_in_frames,
        "predictor": ml_collections.ConfigDict({
          "module": "src.models.ProcModule",
          "z_a_size": z_a_size,
          "z_b_size": z_b_size,
          "z_c_size": z_c_size,
          "D_module": ml_collections.ConfigDict({
                "module": "src.models.TransformerDynamicsPredictor",
                "d_model": slot_size,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 512,
                "norm_first": True,
                "num_context_frames": num_burn_in_frames,
                "input_dim": slot_size,
                "output_dim": 64,
              }),
          "F_module": ml_collections.ConfigDict({
                  "module": "src.models.PhysicsEngine",
                  "learned_parameters": ['G', 'obj_mass'],
                  "G": sample_metadata['flags']['gravitational_constant'],
                  "step_rate": sample_metadata['flags']['step_rate'],
                  "frame_rate": sample_metadata['flags']['frame_rate'],
                  "obj_mass": sample_metadata['flags']['fixed_mass'],
                }),
            "P_in": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "hidden_sizes": (128,),
                "activate_last": True,
            }),
            "P_out": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "hidden_sizes": (128,),
                "activate_last": True,
            }),
            "F_in": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "use_bias": False,
            }),
            "F_out": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "use_bias": False,
            }),
          }),
        "encoder": ml_collections.ConfigDict({
          "module": "src.models.CNNEncoder",
          "norm": '',
          "ks": 5,
          "channels": (3, 64, 64, 64, 64),
          "resolution": resolution,
          "out_size": slot_size,
          "add_pos_emb": True,
          "strides": (1, 1, 1, 1),
          "flatten_output": False
        }),
        "decoder": ml_collections.ConfigDict({
          "module": "src.models.SpatialBroadcastDecoder",
          "input_size": slot_size,
          "resolution": (8, 8),
          "channels": (slot_size, 64, 64, 64, 64),
          "ks": 5,
          "norm": '',
          "strides": (2, 2, 2, 1),
        }),
        "slot_attention": ml_collections.ConfigDict({
          "module": "src.models.SlotAttention",
          "num_iterations": 2,
          "mlp_hidden_size": 256,
        }),
        "state_autoencoder_alignment_factor": 1.0,
        "optimizer": ml_collections.ConfigDict({
                "lr": 2e-4,
                "use_lr_scheduler": False,
        }),
        "learned_parameter_specific_lr": ml_collections.ConfigDict({
                "lr": 1e-2,
        }),
        "train_metrics": {
                "ssim_rgb": functools.partial(SSIM, key='rgb'),
                "psnr_rgb": functools.partial(PSNR, key='rgb'),
        },
        "val_metrics": {
                "ssim_rgb": functools.partial(SSIM, key='rgb'),
                "psnr_rgb": functools.partial(PSNR, key='rgb'),
        },
        "additional_metrics_last_val": {
          # Due to memory requirements only after training is done
          "lpips_rgb": functools.partial(LPIPS, key='rgb'),
        },
  })

  return config
