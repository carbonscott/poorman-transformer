import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import random
import tqdm
import logging

from poorman_transformer.data                 import TokenManager, TinyShakespearDataset
from poorman_transformer.modeling.transformer import Transformer
from poorman_transformer.utils                import init_logger, MetaLog, save_checkpoint, load_checkpoint, set_seed

torch.autograd.set_detect_anomaly(True)

seed = 0
set_seed(seed)

logger = logging.getLogger(__name__)

# [[[ USER INPUT ]]]
timestamp_prev = None # "2023_0505_1249_26"
epoch          = None # 21

drc_chkpt = "chkpts"
fl_chkpt_prev   = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"
path_chkpt_prev = None if fl_chkpt_prev is None else os.path.join(drc_chkpt, fl_chkpt_prev)

timestamp = init_logger(returns_timestamp = True)

input_file_path = "input.txt"

with open(input_file_path, 'r') as fh:
    data = fh.read()
token_lib = sorted(list(set(data)))
size_data  = len(data)
train_frac = 0.9
train_data = data[:int(size_data * train_frac)]
val_data   = data[int(size_data * train_frac):]

token_manager  = TokenManager(token_lib)
context_length = 32
batch_size     = int(1e4)
sample_size    = int(1e5)
num_workers    = 16

dataset_train    = TinyShakespearDataset(data_source = train_data, context_length = context_length, sample_size = sample_size)
dataset_validate = TinyShakespearDataset(data_source = val_data  , context_length = context_length, sample_size = sample_size)
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                shuffle     = True,
                                                pin_memory  = True,
                                                batch_size  = batch_size,
                                                num_workers = num_workers, )
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                shuffle     = True,
                                                pin_memory  = True,
                                                batch_size  = batch_size,
                                                num_workers = num_workers, )

# Define model
token_lib_size = len(token_lib)
embd_size      = 64
num_blocks     = 4
head_size      = 64 // 4
num_heads      = embd_size // head_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Transformer(token_lib_size, embd_size, context_length, num_blocks, num_heads).to(device)
logger.info(f'{sum(p.numel() for p in model.parameters())/1e6}, M parameters')

criterion = nn.CrossEntropyLoss()

lr = 1e-3
weight_decay = 1e-4
param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
## scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
##                                          factor         = 2e-1,
##                                          patience       = 10,
##                                          threshold      = 1e-4,
##                                          threshold_mode ='rel',
##                                          verbose        = True)
scheduler = None


# [[[ TRAIN LOOP ]]]
max_epochs = 5000

# From a prev training???
epoch_min = 0
loss_min  = float('inf')
if path_chkpt_prev is not None:
    epoch_min, loss_min = load_checkpoint(model, optimizer, scheduler, path_chkpt_prev)
    ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
    epoch_min += 1    # Next epoch
    logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

logger.info(f"Current timestamp: {timestamp}")

uses_mixed_precision = True
chkpt_saving_period  = 10
epoch_unstable_end  = 1000
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch += epoch_min

    # Uses mixed precision???
    if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

    # ___/ TRAIN \___
    # Turn on training related components in the model...
    model.train()

    # Fetch batches...
    train_loss_list = []
    batch_train = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train), disable=True)
    for batch_idx, batch_entry in batch_train:
        # Unpack the batch entry and move them to device...
        batch_input, batch_target = batch_entry

        batch_input = torch.tensor([ token_manager.encode(context) for context in batch_input ])
        batch_input = batch_input.to(device)

        # Get target tensor...
        batch_target = torch.tensor([ token_manager.encode(target) for target in batch_target ])
        batch_target = batch_target.to(device)

        # Forward, backward and update...
        if uses_mixed_precision:
            with torch.cuda.amp.autocast(dtype = torch.float16):
                # Forward pass...
                batch_output = model(batch_input)

                # Calculate the loss...
                B, T, N = batch_output.shape
                loss = criterion(batch_output.view(B * T, N), batch_target.view(B * T))
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass...
            batch_output = model(batch_input)

            # Calculate the loss...
            B, T, N = batch_output.shape
            loss = criterion(batch_output.view(B * T, N), batch_target.view(B * T))
            loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting...
        train_loss_list.append(loss.item())

    train_loss_mean = torch.mean(torch.tensor(train_loss_list))
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {train_loss_mean:.8f}")


    # ___/ VALIDATE \___
    model.eval()

    # Fetch batches...
    validate_loss_list = []
    batch_validate = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate), disable = True)
    for batch_idx, batch_entry in batch_validate:
        # Unpack the batch entry and move them to device...
        batch_input, batch_target = batch_entry

        batch_input = torch.tensor([ token_manager.encode(context) for context in batch_input ])
        batch_input = batch_input.to(device)

        # Get target tensor...
        batch_target = torch.tensor([ token_manager.encode(target) for target in batch_target ])
        batch_target = batch_target.to(device)

        # Forward only...
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    # Forward pass...
                    batch_output = model(batch_input)

                    # Calculate the loss...
                    B, T, N = batch_output.shape
                    loss = criterion(batch_output.view(B * T, N), batch_target.view(B * T))
                    loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus
            else:
                # Forward pass...
                batch_output = model(batch_input)

                # Calculate the loss...
                B, T, N = batch_output.shape
                loss = criterion(batch_output.view(B * T, N), batch_target.view(B * T))
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

        # Reporting...
        validate_loss_list.append(loss.item())

    validate_loss_mean = torch.mean(torch.tensor(validate_loss_list))
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {validate_loss_mean:.8f}")

    # Report the learning rate used in the last optimization...
    lr_used = optimizer.param_groups[0]['lr']
    logger.info(f"MSG (device:{device}) - epoch {epoch}, lr used = {lr_used}")

    # Update learning rate in the scheduler...
    if scheduler is not None: scheduler.step(validate_loss_mean)


    # ___/ SAVE CHECKPOINT??? \___
    if validate_loss_mean < loss_min:
        loss_min = validate_loss_mean

        if (epoch % chkpt_saving_period == 0) or (epoch > epoch_unstable_end):
            fl_chkpt   = f"{timestamp}.epoch_{epoch}.chkpt"
            path_chkpt = os.path.join(drc_chkpt, fl_chkpt)
            save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
            logger.info(f"MSG (device:{device}) - save {path_chkpt}")


    # Shuffle the dataset...
    dataset_train.update_random_dataset()
    dataset_validate.update_random_dataset()
