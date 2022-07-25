##################################################
# Imports
##################################################

from ast import arg
import sys
import pytorch_lightning as pl
import logging
import os
import subprocess
import torch

# Custom
from config import get_args
from datasets.dataloaders import get_datasets, get_dataloaders
from Models import relcnn, relvit
from utils import get_logger, get_callbacks
from trainer import get_trainer
import Analysis

# Settings
format_str = '%(asctime)s - [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=format_str, datefmt='%d-%b-%y %H:%M:%S')


# Main function
def main(args):
    # Logging
    if args.num_gpus > 0:
        nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE).communicate()[0].decode('utf8')
        logging.info(nvidia_smi)
    else:
        logging.info('Using CPU!')
    hostname = os.uname().nodename # Node name
    logging.info(f'hostname:{hostname}')
    
    # Dataloaders
    dls = get_dataloaders(args)
    # Get model
    if args.backbone == 'vit':
        model = relvit.get_model(args)
    else:
        if args.backbone =='cifar10_model':
            model = relcnn.get_model(args)
        else:
            model = relcnn.get_model(args)
    # elif args.backbone == 'simsiam':
    #     model = relvit_simsiam.get_model(args)
    # elif args.bacbkone == 'cnn':
    #     model = relcnn.get_model(args)

    # Trainer
    trainer = get_trainer(args, dls)

    # Mode
    if args.mode in ['train', 'training']:
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.validate(model=None, dataloaders=dls['validation'])
        
    elif args.mode in ['validate', 'validation', 'validating']:
        trainer.validate(model, dataloaders=dls['validation'])
    
    elif args.mode in ['train_linear']:
        trainer.fit(model, dls['train_aug'], dls['validation'])
        trainer.validate(model, dataloaders=dls['validation'])

    elif args.mode in ['validate_lienar', 'validation_linear', 'validating_linear']:
        trainer.validate(model, dataloaders=dls['validation'])

    elif args.mode == 'train_simsiam':
        trainer.fit(model, dls['train_simsiam'], dls['validation'])
        trainer.validate(model=None, val_dataloaders=dls['validation'])
    
    elif args.mode == 'test':
        trainer.test(dataloaders=dls['test'])
    
    elif args.mode == 'analysis':
        Analysis.Create(model, dls['validation'], args.stats_path).get_type(args.type)

    else:
        raise Exception(f'Error. Model "{args.mode}" not supported.')


##################################################
# Main
##################################################

if __name__ == '__main__':

    # Parse input arguments
    args = get_args(sys.stdin, verbose=True)
    # Set random seed
    pl.seed_everything(args.seed)

    # Run main
    main(args)

