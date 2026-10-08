"""
Implements Incremental Learning procedure.
"""
import os

# Set environment variables before PyTorch initialization
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Safely configure CUDA memory allocation settings
try:
    import torch
    # Can only be set before PyTorch initialization
    if not torch.cuda.is_initialized():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get('PDR_ALLOC', 'max_split_size_mb:512')
except:
    pass

import logging
import traceback
import json
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size

from utils.config import get_params
from utils.factory import get_model
from utils.dataset import get_dataset

import time
import numpy as np
import random
import wandb

from utils.logger import init_experiment
from utils.dataset import get_dataset


def main_cl(params):
    
    # ------------------------------------------------------------------------------------------------------------------------=====
    
    # Initialize Accelerator
    accelerator = Accelerator(log_with="wandb")
    
    # Using Fixed Random Seed
    if params.seed:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
        if os.environ.get("PDR_DETERMINISTIC", "") == "1":
            # atomicAdd-based kernels (index_add_, scatter_add_, the embedding backward) sum in a run-dependent
            # order, and continual learning amplifies such bit-level differences across tasks. Request
            # deterministic kernels for reproducible runs.
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.cuda.manual_seed_all(params.seed)

    # Initialize Experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)

    # Initialize wandb
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    # retry request (handles connection errors, timeouts)
    try_cnt = 0
    while try_cnt<5:
        try:
            accelerator.init_trackers(
                project_name=params.wandb_project,
                init_kwargs={
                    "wandb":{
                        'entity':params.wandb_entity, 
                        'name':params.wandb_name,
                        'config':vars(params), 
                        'settings':wandb.Settings(start_method="fork"),
                        # Disable wandb when debug
                        'mode': 'disabled' if 'default' in params.exp_prefix else 'online' if params.is_wandb else 'offline'
                    }
                }
            )
            break
        except Exception as e:
            print(str(e))
            print("Retrying Connecting wandb...")
            try_cnt+=1
            time.sleep(120)

    # Dataset
    CL_dataset = get_dataset(params)

    # Start with a smaller batch size for large models
    if params.backbone == 'gpt-oss-20b':
        starting_batch_size = 1
    else:
        starting_batch_size = params.batch_size

    # Conditionally apply @find_executable_batch_size
    # Automatic batch-size reduction on OOM is opt-in via PDR_BS_FALLBACK=1.
    if params.backbone != 'meta-llama/Llama-3.2-1B' and os.environ.get('PDR_BS_FALLBACK', '') == '1':
        @find_executable_batch_size(starting_batch_size=starting_batch_size)
        def inner_training_loop(batch_size):
            nonlocal accelerator
            
            # More aggressive memory cleanup
            accelerator.free_memory()
            if hasattr(accelerator, 'clear_memory'):
                accelerator.clear_memory()
            
            # PyTorch memory cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            params.__setattr__('batch_size', batch_size)
            model = get_model(params, CL_dataset, accelerator)
            model.incremental_training()
            model.finish_training()

        inner_training_loop()
    else:
        def inner_training_loop():
            nonlocal accelerator
            
            # More aggressive memory cleanup
            accelerator.free_memory()
            if hasattr(accelerator, 'clear_memory'):
                accelerator.clear_memory()
            
            # PyTorch memory cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            model = get_model(params, CL_dataset, accelerator)
            model.incremental_training()
            model.finish_training()

        inner_training_loop()
        
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    params = get_params()
    main_cl(params)
