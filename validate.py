import os
import sys
import pwd
import importlib
import argparse
import pathlib
import logging
import tempfile
import ml_collections

from absl import app
from absl import flags

from dotenv import load_dotenv
from ml_collections import config_flags

import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

from src.datasets import ViProDataModule
from src.lib import get_module



FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Config file.")
flags.DEFINE_integer('random_seed', 41, "Random Seed")
flags.DEFINE_string('run_id', 'undefined', "Run Name")
flags.DEFINE_integer('num_val_frames', -1, "Number of validation frames")
flags.DEFINE_integer('batch_size', 64, "Batch Size")

def validate(config, run_id, print_only, num_val_frames, batch_size):
    rank_zero_info(f"sys.version: {sys.version}")
    rank_zero_info('Command: python ' + ' '.join(sys.argv))

    pl.seed_everything(config.seed)

    with config.unlocked():
        if num_val_frames != -1:
            config.data.num_val_frames = num_val_frames

    data_config = config.data.to_dict()
    data_config['batch_size'] = batch_size

    # Setup data
    data_module = ViProDataModule(**data_config, random_seed=config.seed, train_fraction=1.0, max_instances=config.num_slots_override - 1 if 'num_slots_override' in config else None)

    with tempfile.TemporaryDirectory() as tmpdirname:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path='logs/checkpoints', dst_path=tmpdirname)

        checkpoint_path = os.path.join(tmpdirname, 'logs', 'checkpoints')

        additional_config = {}
        if num_val_frames != -1 and 'num_val_rollout_frames' in config.model:
            additional_config['num_val_frames'] = num_val_frames

        if 'test_settings' in config.model:
            additional_config['test_settings'] = config.model['test_settings']

        model = get_module(config.model.module).load_from_checkpoint(
            os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]),
            **additional_config, strict=False, val_metrics=config.model.val_metrics.to_dict(),
            additional_metrics_last_val=config.model.additional_metrics_last_val.to_dict())

    callbacks = []

    callbacks.append(pl.callbacks.RichProgressBar())

    trainer = pl.Trainer(
        callbacks=callbacks, accelerator="gpu", devices=1,
        enable_progress_bar=True,
        max_epochs=1, num_nodes=1,
        gradient_clip_val=config.max_grad_norm if 'max_grad_norm' in config else None,
        precision=config.precision if 'precision' in config else 32)

    model.setup_final_validation()
    rank_zero_info("Obtaining best validation metrics")
    best_metrics = trainer.validate(model, data_module)

    if not print_only:
        rank_zero_info("Logging best metrics to mlflow")

        for metrics in best_metrics:
            mlflow.log_metrics(metrics, step=trainer.current_epoch)

        mlflow.end_run()

def main(argv):
    del argv

    config = FLAGS.config
    num_val_frames = FLAGS.num_val_frames
    print_only = FLAGS.print_only
    batch_size = FLAGS.batch_size
    run_id = FLAGS.run_id

    if type(config) == ml_collections.config_flags.config_flags._ErrorConfig:
        # Load config from checkpoint
        with tempfile.TemporaryDirectory() as tmpdirname:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path='', dst_path=tmpdirname)

            config_path = os.path.join(tmpdirname, 'config')
            config_path = os.path.join(config_path, os.listdir(config_path)[0])
            print(config_path)

            config_file = importlib.machinery.SourceFileLoader('config_file', config_path).load_module()
            config = config_file.get_config()

    with config.unlocked():
        config.seed = FLAGS.random_seed

    validate(config=config, run_id=run_id, print_only=print_only, num_val_frames=num_val_frames, batch_size=batch_size)

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.resolve())
    load_dotenv(pathlib.Path(__file__).parent.joinpath('.env').resolve())

    app.run(main)
