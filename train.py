import os
import sys
import pathlib
import logging

from absl import app
from absl import flags

from dotenv import load_dotenv
from ml_collections import config_flags

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

from src.lib import init_module
from src.datasets import ViProDataModule

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Config file.")
flags.DEFINE_integer('random_seed', 41, "Random Seed")
flags.DEFINE_string('run_name', 'undefined', "Run Name")
flags.DEFINE_float('train_fraction', 1.0, "Fraction of training set to use")
flags.DEFINE_integer('batch_size', -1, "Batch Size")

flags.mark_flags_as_required(["config"])

@rank_zero_only
def init_logger(experiment_name, run_name, resume=None):
    log_dir = f'logs/{experiment_name}/{run_name}'

    if os.path.exists(log_dir):
        index = 1
        adjusted_log_dir = f'{log_dir}_{index}'
        while os.path.exists(adjusted_log_dir):
            index += 1
            adjusted_log_dir = f'{log_dir}_{index}'

        log_dir = adjusted_log_dir

    os.makedirs(log_dir, exist_ok=False)

    # Logging Configuration
    formatter = logging.Formatter(
      fmt='%(asctime)s [%(levelname)s] : %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S')

    logging.getLogger().setLevel(logging.INFO)

    logging.getLogger('pytorch_lightning').setLevel(logging.INFO)

    if log_dir:
        if resume:
            fh = logging.FileHandler(os.path.join(log_dir, 'output.log'), mode='a')
        else:
            fh = logging.FileHandler(os.path.join(log_dir, 'output.log'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)
        logging.getLogger('pytorch_lightning').addHandler(fh)

    if resume and log_dir:
        logging.info("-------------------------")
        logging.info("Appending to existing log")
        logging.info("-------------------------")

    return log_dir

def train(config, log_dir, batch_size):
    rank_zero_info(f"sys.version: {sys.version}")
    rank_zero_info('Command: python ' + ' '.join(sys.argv))
    rank_zero_info(f"Logging to: {log_dir}")

    pl.seed_everything(config.seed)

    data_config = config.data.to_dict()
    batch_size = data_config['batch_size'] if batch_size is None else batch_size

    data_config['batch_size'] = batch_size
    early_stopping = config.get('early_stopping', False)
    early_stopping_patience = config.get('early_stopping_patience', 50)
    early_stopping_monitor = config.get('early_stopping_monitor', 'val/loss')
    check_val_every_n_epoch = config.get('check_val_every_n_epoch', 1)

    # Setup data
    if 'module' in data_config:
        data_module = init_module(data_config, random_seed=config.seed, train_fraction=config.train_fraction, max_instances=config.num_slots_override - 1 if 'num_slots_override' in config else None)
    else:
        data_module = ViProDataModule(**data_config, random_seed=config.seed, train_fraction=config.train_fraction, max_instances=config.num_slots_override - 1 if 'num_slots_override' in config else None)

    # Infer number of epochs from num_steps
    num_epochs = config.num_steps // (len(data_module.train_data) // (data_module.batch_size))

    # Setup model
    model = init_module(config.model, total_steps=config.num_steps,
                        num_val_frames=config.data.num_val_frames,
                        num_slots=config.num_slots_override if 'num_slots_override' in config else data_module.max_instances + 1)

    logger = [pl.loggers.TensorBoardLogger(log_dir, name='', version='.', log_graph=False)]

    callbacks = []

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=early_stopping_monitor,
            mode='min', save_top_k=1, every_n_epochs=None,
        dirpath=os.path.join(log_dir, 'checkpoints'))

    if early_stopping:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor=early_stopping_monitor, patience=early_stopping_patience, mode='min')
        callbacks.append(early_stop_callback)

    callbacks.append(checkpoint_callback)

    callbacks.append(pl.callbacks.ModelSummary(max_depth=50))

    callbacks.append(pl.callbacks.RichProgressBar())

    pl.Trainer()

    trainer = pl.Trainer(
        logger=logger, callbacks=callbacks, accelerator="gpu", devices=1,
        enable_progress_bar=True,
        max_epochs=num_epochs, num_nodes=1,
        gradient_clip_val=config.max_grad_norm if 'max_grad_norm' in config else None,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision=config.precision if 'precision' in config else 32)

    trainer.fit(model, data_module)

def main(argv):
    del argv
    config_param = ''
    for v in sys.argv:
        if v.startswith('--config'):
            config_param = v.split('=')[1]

    if not config_param:
        raise ValueError("cfg file not specified or in invalid format. it needs to be --config=CONFIG")

    config = FLAGS.config
    with config.unlocked():
        config.seed = FLAGS.random_seed
        if 'train_fraction' not in config:
            config.train_fraction = FLAGS.train_fraction

    run_name = FLAGS.run_name
    batch_size = FLAGS.batch_size
    if batch_size == -1:
        batch_size = None

    log_dir = init_logger(experiment_name=config.experiment_name, run_name=run_name, resume=None)
    if not log_dir:
        # Not rank zero, does not matter
        log_dir = '.'

    train(config=config, log_dir=log_dir,
          batch_size=batch_size)

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.resolve())
    load_dotenv(pathlib.Path(__file__).parent.joinpath('.env').resolve())

    app.run(main)
