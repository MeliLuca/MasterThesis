##################################################
# Imports
##################################################

from cgitb import enable
import pytorch_lightning as pl

# Custom
from utils import get_logger, get_callbacks


def get_trainer(args, dls):
    """
    Return the PyTorch Lightning Trainer.
    """

    # Logger and callbacks
    logger = get_logger(args)
    callbacks = get_callbacks(args)

    # Trainer
    trainer_args = {
        'gpus': args.num_gpus,
        'max_epochs': args.epochs,
        'deterministic': True,
        'callbacks': callbacks,
        'logger': logger,
        'gradient_clip_val': args.grad_clip_val,
        'max_steps': args.epochs * len(dls['train_aug']),
        'enable_progress_bar': args.enable_progress_bar,
    }
    trainer = pl.Trainer(**trainer_args)
    return trainer