import logging
import os

# Import from third library
import torch
import torch.distributed as dist

logger = logging.getLogger('global')

SyncBatchNorm2d = torch.nn.SyncBatchNorm
# logger.info("You are using fake SyncBatchNorm2d who is actually the official BatchNorm2d")

