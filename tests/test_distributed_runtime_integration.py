"""DataLoader/Gloo lifecycle smoke, isolated with an external test timeout.

Run directly or via unittest discovery. Set CDFSOD_SHUTDOWN_BACKEND=nccl
on the training server to exercise the same lifecycle with CUDA/NCCL.
This does not run the detector or MMEngine checkpoint hooks.
"""
import importlib.util
import ipaddress
import os
from pathlib import Path
import subprocess
import socket
import struct
import sys
import tempfile
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        'distributed_runtime', ROOT / 'mmdet/utils/distributed_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.distributed_runtime


def assert_loopback_listeners():
    """Inspect actual native listeners, before creating auxiliary Gloo groups."""
    inodes = set()
    for fd in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            inodes.add(target[8:-1])
    for table, family in (('tcp', socket.AF_INET), ('tcp6', socket.AF_INET6)):
        for row in Path(f'/proc/self/net/{table}').read_text().splitlines()[1:]:
            fields = row.split()
            if fields[3] != '0A' or fields[9] not in inodes:
                continue
            host = fields[1].split(':')[0]
            packed = b''.join(struct.pack('!I', int(host[i:i + 8], 16))
                              for i in range(0, len(host), 8))
            # /proc prints each address word in host byte order.
            packed = b''.join(packed[i:i + 4][::-1] for i in range(0, len(packed), 4))
            address = ipaddress.ip_address(socket.inet_ntop(family, packed))
            address = getattr(address, 'ipv4_mapped', None) or address
            assert address.is_loopback, f'Native listener exposed on {address}: {row}'


def lifecycle(directory, mode):
    runtime = load_runtime()
    backend = os.environ.get('CDFSOD_SHUTDOWN_BACKEND', 'gloo')
    if backend == 'nccl':
        torch.cuda.set_device(0)
    device = 'cuda:0' if backend == 'nccl' else 'cpu'
    runner = SimpleNamespace()
    workers = []
    original_destroy = dist.destroy_process_group

    def destroy_after_workers():
        assert workers and all(not worker.is_alive() for worker in workers)
        original_destroy()

    try:
        with patch.object(dist, 'destroy_process_group', destroy_after_workers):
            with runtime(lambda: runner):
                store = dist.TCPStore('127.0.0.1', 0, world_size=1, is_master=True,
                                      use_libuv=False)
                dist.init_process_group(backend, store=store, rank=0, world_size=1,
                                        timeout=timedelta(seconds=30))
                value = torch.tensor([1.], device=device)
                dist.all_reduce(value)
                if backend == 'nccl' and sys.platform == 'linux':
                    assert os.environ['NCCL_SOCKET_IFNAME'] == '=lo'
                    assert_loopback_listeners()
                # Also verify auxiliary group cleanup, as used by MMEngine.
                auxiliary = dist.new_group(backend='gloo')
                dist.all_reduce(torch.ones(1), group=auxiliary)
                phases = (('train', 4), ('val', 2)) if mode != 'test' else (('test', 2),)
                for phase, count in phases:
                    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2,
                                        num_workers=count, persistent_workers=(mode != 'test'))
                    setattr(runner, f'_{phase}_loop', SimpleNamespace(dataloader=loader))
                    iterator = iter(loader)
                    workers.extend(iterator._workers)
                    batches = list(iterator)
                    assert torch.equal(torch.cat([batch[0] for batch in batches]), torch.arange(8))
                # Persistent workers really are still running after exhaustion.
                assert all(worker.is_alive() == (mode != 'test') for worker in workers)
                if mode == 'failure':
                    raise ValueError('expected training failure')
                checkpoint = directory / 'best.pth'
                if mode == 'train':
                    torch.save({'completed_validation': True, 'value': value.cpu()}, checkpoint)
                else:
                    saved = torch.load(checkpoint, weights_only=True)
                    assert saved['completed_validation']
                    assert torch.equal(saved['value'], value.cpu())
                    (directory / 'test.completed').write_text('ok', encoding='utf-8')
    except ValueError as error:
        assert mode == 'failure' and str(error) == 'expected training failure'
    assert not dist.is_initialized()
    assert workers and all(not worker.is_alive() for worker in workers)
    print(f'{mode}: workers stopped, groups destroyed, process returning', flush=True)


class DistributedRuntimeIntegrationTests(unittest.TestCase):
    def run_stage(self, directory, mode):
        env = dict(os.environ, WORLD_SIZE='1')
        result = subprocess.run([sys.executable, __file__, '--stage', str(directory), mode],
                                capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('process returning', result.stdout)
        self.assertIn('Distributed process groups destroyed.', result.stderr)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'requires Gloo')
    def test_training_shutdown_allows_checkpoint_test_in_next_process(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.run_stage(directory, 'train')
            self.assertTrue((directory / 'best.pth').is_file())
            self.run_stage(directory, 'test')
            self.assertEqual((directory / 'test.completed').read_text('utf-8'), 'ok')

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'requires Gloo')
    def test_training_failure_releases_workers_and_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            self.run_stage(Path(temp), 'failure')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--stage':
        lifecycle(Path(sys.argv[2]), sys.argv[3])
    else:
        unittest.main()
