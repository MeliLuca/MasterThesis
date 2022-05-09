import sys
import pytorch_lightning as pl
import logging
import os
import subprocess
from config import get_args

from Dataloaders import get_dataloaders
import Models


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
        'max_steps': args.epochs * len(dls['train'])
    }
    trainer = pl.Trainer(**trainer_args)
    return trainer


def main (args):

    dls = get_dataloaders(args)
    trainer = get_trainer(args,dls)

    if args.model == 'resnet':
        if args.dataset_name == 'cifar10':
            model = Models.ResNetCifar10(args)
   

    # Mode
    if args.mode in ['train', 'training']:
        trainer.fit(model, dls['train'], dls['validation'])
        trainer.validate(model=None, val_dataloaders=dls['validation'])

    elif args.mode in ['validate', 'validation', 'validating']:
        trainer.validate(model, val_dataloaders=dls['validation'])

    elif args.mode in ['self_supervise']:
        
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.validate(model, val_dataloaders=dls['validation'])

    else:
        raise Exception(f'Error. Model "{args.mode}" not supported.')

##################################################
# Main
##################################################

if __name__ == '__main__':

    # Parse input arguments
    args = get_args(sys.stdin, verbose=True)

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    # Run main
    main(args)

