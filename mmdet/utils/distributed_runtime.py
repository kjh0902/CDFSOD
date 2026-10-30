# Copyright (c) OpenMMLab. All rights reserved.
"""Explicit worker and process-group lifetime for the CLI tools."""
import os
import sys
import traceback
from contextlib import contextmanager

import torch.distributed as dist


def _configure_single_rank_nccl():
    """Keep torchrun's single-rank NCCL sockets off public interfaces.

    MASTER_ADDR only controls the rendezvous store, not NCCL's proxy listener.
    A single rank needs no network peers. NCCL 2.26.2 can block forever accepting
    an incomplete handshake from an unrelated remote connection, including on
    shutdown. Select the exact Linux loopback interface before NCCL initializes.
    Leave multi-rank jobs and already initialized groups alone.
    """
    if (sys.platform == 'linux' and os.environ.get('WORLD_SIZE') == '1'
            and dist.is_available() and not dist.is_initialized()):
        os.environ['NCCL_SOCKET_IFNAME'] = '=lo'
        print('[distributed] Single-rank NCCL socket interface: =lo (loopback).',
              file=sys.stderr, flush=True)


def _shutdown_runner_workers(runner):
    """Stop existing DataLoader iterators without building unused loops.

    Persistent workers survive iterator exhaustion and can inherit distributed
    resources under fork. Release them before shutting down process groups.
    PyTorch has no public DataLoader.close(); use the iterator's own shutdown
    routine so pin-memory threads, worker queues and workers close in order.
    """
    seen = set()
    errors = []
    for phase in ('train', 'val', 'test'):
        # Runner's public loop/dataloader properties build objects lazily.
        loop = getattr(runner, f'_{phase}_loop', None)
        if loop is None or isinstance(loop, dict):
            continue
        loader = getattr(loop, 'dataloader', None)
        # IterBasedTrainLoop also retains a non-persistent iterator here.
        infinite_iterator = getattr(loop, 'dataloader_iterator', None)
        iterators = (getattr(loader, '_iterator', None),
                     getattr(infinite_iterator, '_iterator', None))
        for iterator in iterators:
            if iterator is None or id(iterator) in seen:
                continue
            seen.add(id(iterator))
            shutdown = getattr(iterator, '_shutdown_workers', None)
            if not callable(shutdown):
                continue
            print(f'[shutdown] Stopping {phase} DataLoader workers.',
                  file=sys.stderr, flush=True)
            try:
                shutdown()
            except Exception as error:
                errors.append(error)
                traceback.print_exc()
            else:
                print(f'[shutdown] {phase} DataLoader workers stopped.',
                      file=sys.stderr, flush=True)
    if errors:
        raise errors[0]


@contextmanager
def distributed_runtime(runner_getter=None):
    """Release runner workers, then process groups, before Python shutdown.

    ``runner_getter`` is evaluated only on exit, allowing construction failures
    to use the same cleanup path. Enclose runner construction as well: MMEngine
    can initialize communication before model/checkpoint loading fails.
    Do not add a barrier here; another rank may already have failed.
    """
    _configure_single_rank_nccl()
    try:
        yield
    finally:
        # Preserve the original training/testing exception if cleanup also fails.
        failed = sys.exc_info()[0] is not None
        try:
            if runner_getter is not None:
                _shutdown_runner_workers(runner_getter())
        except Exception:
            if not failed:
                raise
            print('[shutdown] DataLoader cleanup also failed:',
                  file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            # Attempt group cleanup even if stopping a worker raised an error.
            failed = sys.exc_info()[0] is not None
            if dist.is_available() and dist.is_initialized():
                print('[shutdown] Destroying distributed process groups.',
                      file=sys.stderr, flush=True)
                try:
                    # No group argument: include auxiliary MMEngine Gloo groups.
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
