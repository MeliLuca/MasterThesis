import sys
import pytorch_lightning as pl
import logging
import os
import subprocess
from config import get_args

from Dataloaders import get_dataloaders
from Utils import get_callbacks
from Models import ResNetCifar10


def get_trainer(args, dls):
    """
    Return the PyTorch Lightning Trainer.
    """

    # Logger and callbacks
    callbacks = get_callbacks(args)

    # Trainer
    trainer_args = {
        'gpus': args.num_gpus,
        'max_epochs': args.epochs,
        'deterministic': True,
        'callbacks': callbacks,
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
            model = ResNetCifar10.get_model(args)
   

    # Mode
    if args.mode in ['train', 'training']:
        trainer.fit(model, dls['train'], dls['validation'])
        trainer.validate(model=None, val_dataloaders=dls['validation'])

    elif args.mode in ['validate', 'validation', 'validating']:
        trainer.validate(model, val_dataloaders=dls['validation'])

    elif args.mode in ['self_supervise']:
        trainer.fit(model,dls['train'], dls['validation'], part='multi_attention')
        trainer.fit(model, dls['train'], dls['validation'], part='')
        trainer.validate(model, val_dataloaders=dls['validation'], part='multi_attention')

    else:
        raise Exception(f'Error. Model "{args.mode}" not supported.')
    
    trainer.test(dataloaders=dls['test'])

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

