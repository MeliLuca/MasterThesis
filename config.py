##################################################
# Imports
##################################################

import argparse
import json
import logging


def get_args(stdin, verbose=False):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(stdin)

    # Global params
    parser.add_argument('--seed', type=int, default=35771, help='The random seed.')
    parser.add_argument('--logger', type=str, default='csv', help='The logger to use for the experiments.')
    parser.add_argument('--mode', type=str, default='train', help='The mode of the program, can "train" or "validate"')
    parser.add_argument('--num_gpus', type=int, default=1, help='The number of GPUs.')
    parser.add_argument('--epochs', type=int, default=190, help='The number of epochs for the training.')
    parser.add_argument('--task', type=str, default='downstream', 
                        help='The name of the task, can be "upstream" or "downstream"')
    parser.add_argument('--exp_id', type=str, default='', 
                        help='Experiment ID (like "u1").')
    parser.add_argument('--enable_progress_bar', type=bool, default=True, help='Show the progress bar')

    # Datasets params
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset.')
    parser.add_argument('--data_base_path', type=str, default='./datasets/data', help='The data base path.')
    parser.add_argument('--stats_path', type=str, default='', help='The statistics path (loggers and checkpoints).')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers for the dataloader.')
    parser.add_argument('--debug_dataset', dest='debug_dataset', action='store_true', 
                        help='Print useful info about the dataset.')
    parser.add_argument('--shuffle_patches', dest='shuffle_patches', action='store_true', 
                        help='Shuffle the input patches.')

    # Optimizer params
    parser.add_argument('--optimizer', type=str, default='adamw', help='The kind of optimizer.', 
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4, help='The learning rate.')
    parser.add_argument('--metric_monitor', type=str, default='val_acc', 
                        help='The metric used for early stopping.')
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help='Gradient clipping value.')

    # Model params
    parser.add_argument('--backbone', type=str, default='resnet', help='Backbone of the model.')
    parser.add_argument('--model_size', type=str, default='small', help='Backbone of the model.')
    parser.add_argument('--model_checkpoint', type=str, default='', help='The model checkpoint path (*.ckpt).')
    parser.add_argument('--patch_size', type=int, default=-1, help='The patch size.')
    parser.add_argument('--use_relations', dest='use_relations', action='store_true', 
                        help='Learn patches spatial relations.')
    parser.add_argument('--use_supervision', dest='use_supervision', action='store_true', help='Learn image classes.')
    parser.add_argument('--use_distances', dest='use_dist', action='store_true', help='Learn patch distances.')
    parser.add_argument('--use_angles', dest='use_angles', action='store_true', help='Learn patch angles.')
    parser.add_argument('--use_abs_positions', dest='use_abs_positions', action='store_true', 
                        help='Learn absolute positions of patches.')
    parser.add_argument('--use_attention', dest='use_attention', action='store_true', help='learn with torch standard attention layer')
    parser.add_argument('--use_positional_embeddings', dest='use_positional_embeddings', action='store_true', 
                        help='Use positional embeddings in the ViT.')
    parser.add_argument('--freeze_fet_enc', dest='freeze_fet_enc', action='store_true', 
                        help='Freeze backbone weights.')
    parser.add_argument('--dropout', type=float, default=0.1, help='The dropout rate of the model.')
    parser.add_argument('--dropout_rel', type=float, default=0.0, 
                        help='The dropout rate applied to the spatial relation classification.')
    parser.add_argument('--smooth_rel', type=float, default=0.0, 
                        help='The label smoothing applied to the spatial relation classification.')
    parser.add_argument('--attn_drop', type=float, default=0.0, 
                        help='Dropout int he attention mask of the model.')
    parser.add_argument('--attn_mask', type=str, default='', 
                        help='Attention mask of the model. Can be empty for standard attention, "diagonal" and "diagonal_clf_row"')
    parser.add_argument('--add_clf_to_patches', dest='add_clf_to_patches', action='store_true', 
                        help='Add CLF token to the patches representation.')
    parser.add_argument('--mega_patches', dest='mega_patches', action='store_true', 
                        help='Use megapatches.')
    parser.add_argument('--side_megapatches', type=int, default=5, 
                        help='Output side of images when using the mega-patches.')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Dropout path rate fo rht ViT.')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Dropout rate of linear proj.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay of the optimizer.')
    parser.add_argument('--pretrained', type=str, default=False, help='Weight decay of the optimizer.')
    parser.add_argument('--emb_dim', type=str, default=64, help='Embedded dim for attention layer.')


    # Parse args
    args = parser.parse_args()
    num_classes = {
        'cifar10': 10,
        'cifar10_36': 10,
        'cifar10_226': 10,
        'cifar10_224': 10,
        'cifar100': 100,
        'cifar100_224': 100,
        'tiny_imagenet': 200,
        'voc': 20,
        'imagenet': 1000,
        'caltech256': 257,
        'flower102': 102,
        'oxford_pet': 37,
        'tiny_imagenet_224': 200,
        'cars': 196,
        'svhn': 10,
    }
    args.num_classes = num_classes[args.dataset]
    # if args.task == 'downstream':
    #     args.metric_monitor = 'val_acc_rel'
    #     args.metric_maxmin = 'max'
    # else:
    #     args.metric_monitor = 'val_loss'
    #     args.metric_maxmin = 'min'

    # Patch size
    if args.patch_size == -1:
        patch_size = {
            'cifar10': 4,
            'cifar10_36': 4,
            'cifar10_226': 64,
            'cifar10_224': 49,
            'cifar100': 4,
            'cifar100_224': 32,
            'tiny_imagenet': 8,
            'tiny_imagenet_224': 32,
            'imagenet': 49,
            'caltech256': 32, 
            'flower102': 32,
            'oxford_pet': 32,
            'cars': 32,
            'svhn': 4,
        }
        args.patch_size = patch_size[args.dataset]
    img_height = {
        'cifar10': 32,
        'cifar10_36': 36,
        'cifar10_226': 226,
        'cifar10_224': 224,
        'cifar100': 32,
        'cifar100_224': 224,
        'tiny_imagenet': 64,
        'tiny_imagenet_224': 224,
        'imagenet': 224,
        'caltech256': 224,
        'flower102': 224,
        'oxford_pet': 224,
        'cars': 224,
        'svhn': 32,
    }
    args.img_height = img_height[args.dataset]
    if verbose:
        args_dict = vars(args)
        logging.info('Input Args: ' + json.dumps({k: args_dict[k] for k in sorted(args_dict)}, indent=4))
    return args
