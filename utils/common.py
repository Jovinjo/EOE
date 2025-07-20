import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import random
import ast
import logging

# Set a fixed seed for reproducibility
def setup_seed(seed=5):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Get label names for visualization or evaluation
def get_test_labels(args, loader=None):
    if args.in_dataset in ['bird200', 'pet37', 'cub100_ID', 'pet18_ID']:
        if loader is None:
            raise ValueError("Loader must be provided for datasets with class_names_str.")
        return loader.dataset.class_names_str
    
    elif args.in_dataset in ['dtd', 'cifar10_ID', 'cifar100_OOD']:
        if loader is None:
            raise ValueError("Loader must be provided for datasets with classes.")
        return loader.dataset.classes
    
    else:
        raise ValueError(f"Unknown dataset for labels: {args.in_dataset}")


# Get number of classes for classification head or metrics
def get_num_cls(args):
    NUM_CLS_DICT = {
        'bird200': 200,
        'pet37': 37,
        'dtd': 47,
        'cub100_ID': 100,
        'cub100_OOD': 100,
        'pet18_ID': 18,
        'pet19_OOD': 19,
        'cifar10_ID': 10,
        'cifar100_OOD': 100,
    }

    try:
        return NUM_CLS_DICT[args.in_dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset: {args.in_dataset}")