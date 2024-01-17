import ml_collections
import os
import functools
import pickle

from src.lib import LPIPS, SSIM, PSNR

def get_config():
  config = ml_collections.ConfigDict()

  config.experiment_name = 'pendulum-camera'

  config.num_steps = 500000
  config.max_grad_norm = 0.05
  config.log_n_samples = 5
  config.early_stopping = True
  config.early_stopping_monitor = "val/loss/post_rgb_recon"

  resolution = (64, 64)

  num_burn_in_frames = 6
  num_train_frames = 12
  num_val_frames = 24

  z_a_size = 64 * 4
  z_b_size = 64 * 4
  z_c_size = 64 * 4

  latent_size = z_a_size + z_b_size + z_c_size

  config.data = ml_collections.ConfigDict({
        "data_dir": os.environ["DATA_DIR"],
        "name": "pendulum-camera",
        "batch_size": 16, # Total
        "num_train_frames": num_train_frames,
        "num_val_frames": num_val_frames,
        "num_burn_in_frames": num_burn_in_frames,
        "load_keys": ["video", "camera"],
        "target_size": resolution,
  })

  with open(os.path.join(config.data.data_dir, config.data.name, '0', 'metadata.pkl'), 'rb') as f:
      sample_metadata = pickle.load(f)

  config.model = ml_collections.ConfigDict({
        "module": "src.models.ProcVIP",
        "do_segmentation": False,
        "num_context_frames": num_burn_in_frames,
        "predictor": ml_collections.ConfigDict({
          "module": "src.models.ProcModule",
          "z_a_size": z_a_size,
          "z_b_size": z_b_size,
          "z_c_size": z_c_size,
          "num_objects": 1,
          "D_module": ml_collections.ConfigDict({
                "module": "src.models.TransformerDynamicsPredictor",
                "d_model": latent_size,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 512,
                "norm_first": True,
                "num_context_frames": num_burn_in_frames,
                "input_dim": latent_size,
                "output_dim": z_b_size,
              }),
          "F_module": ml_collections.ConfigDict({
                  "module": "src.models.CameraPendulum",
                  "link_len": 5.0,
                  "step_rate": sample_metadata['flags']['step_rate'],
                  "frame_rate": sample_metadata['flags']['frame_rate'],
                }),
            "P_in": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "hidden_sizes": (latent_size,),
                "activate_last": True,
            }),
            "P_out": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "hidden_sizes": (latent_size,),
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
          "out_size": latent_size,
          "add_pos_emb": False,
          "strides": (2, 2, 2, 2),
          "flatten_output": True
        }),
        "decoder": ml_collections.ConfigDict({
          "module": "src.models.SpatialBroadcastDecoder",
          "input_size": latent_size,
          "resolution": (8, 8),
          "channels": (latent_size, 64, 64, 64, 64),
          "out_channels": 3,
          "ks": 5,
          "norm": '',
          "strides": (2, 2, 2, 1),
        }),
        "state_autoencoder_alignment_factor": 1.0,

        "optimizer": ml_collections.ConfigDict({
                "lr": 2e-4,
                "use_lr_scheduler": False,
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
          "lpips_rgb": functools.partial(LPIPS, key='rgb'),
        },
  })

  return config
