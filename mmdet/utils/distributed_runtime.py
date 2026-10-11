# Copyright (c) OpenMMLab. All rights reserved.
"""Explicit process-group lifetime for the train/test command-line tools."""
import sys
import traceback
from contextlib import contextmanager

import torch.distributed as dist


@contextmanager
def distributed_runtime():
    """Release process groups before interpreter shutdown, on every rank.

    Enclose runner construction as well: MMEngine can initialize distributed
    communication before model construction or checkpoint loading fails.
    Do not add a barrier here; another rank may already have failed.
    """
    try:
        yield
    finally:
        # Preserve the original training/testing exception if cleanup also fails.
        failed = sys.exc_info()[0] is not None
        if dist.is_available() and dist.is_initialized():
            print('[shutdown] Destroying distributed process groups.',
                  file=sys.stderr, flush=True)
            try:
                # No group argument: destroy all groups, including auxiliary
                # Gloo groups created by MMEngine, even when world_size == 1.
                dist.destroy_process_group()
            except Exception:
                if not failed:
                    raise
                print('[shutdown] Process-group cleanup also failed:',
                      file=sys.stderr, flush=True)
                traceback.print_exc()
            else:
                print('[shutdown] Distributed process groups destroyed.',
                      file=sys.stderr, flush=True)
