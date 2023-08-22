import logging
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor, ceil
from datetime import datetime


logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return None



def init_logger(returns_timestamp = False):
    # Create a timestamp to name the log file...
    now = datetime.now()
    timestamp = now.strftime("%Y_%m%d_%H%M_%S")

    # Configure the location to run the job...
    drc_cwd = os.getcwd()

    # Set up the log file...
    fl_log         = f"{timestamp}.train.log"
    DRCLOG         = "logs"
    prefixpath_log = os.path.join(drc_cwd, DRCLOG)
    if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
    path_log = os.path.join(prefixpath_log, fl_log)

    # Config logging behaviors
    logging.basicConfig( filename = path_log,
                         filemode = 'w',
                         format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                         datefmt="%m/%d/%Y %H:%M:%S",
                         level=logging.INFO, )
    logger = logging.getLogger(__name__)

    return timestamp if returns_timestamp else None




class MetaLog:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ MetaLog \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




def save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path):
    torch.save({
        'epoch'               : epoch,
        'loss_min'            : loss_min,
        'model_state_dict'    : model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
    }, path)




def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    if model     is not None: model.module.load_state_dict(checkpoint['model_state_dict']) \
                              if hasattr(model, 'module') else \
                              model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['loss_min']

