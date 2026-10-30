# NEU-DET 10-shot shutdown investigation (2026-10-30)

Applies to `acl-vistoken-support-memory`, based on `8648db1`.

## Confirmed cause

The user reported all 100 epochs and the last validation completed. The last
messages were `Runner.train() completed; starting shutdown.` and
`[shutdown] Destroying distributed process groups.` at 01:20:16 KST.
The server runs PyTorch 2.7.1+cu126, CUDA 12.6, NCCL 2.26.2 and MMEngine 0.10.7.

The supplied `shutdown_55007.txt` shows this native wait chain:

* Main thread (LWP 55007): `ProcessGroupNCCL::shutdown` ->
  `NCCLComm::destroy` -> `ncclCommDestroy` -> `asyncJobLaunch` -> `usleep`.
* Reclaim thread (LWP 60194): `commReclaim` -> `commCleanup` -> `commFree` ->
  `pthread_join`, waiting for the proxy service thread.
* Proxy thread (LWP 55076): `ncclProxyService` -> `ncclSocketAccept` ->
  `socketFinalizeAccept` -> `socketWait` -> `recv`, requesting 8 bytes on fd 82.

The user's subsequent `ss -tnp` output identifies fd 82 as an established
connection from an external IP to a public-interface port on the training
server. Only the main process owns fd 82. This is **not** a collective waiting
for another training rank, a GIL deadlock, checkpoint I/O, or a DataLoader join.
The native shutdown has reached communicator destruction; the proxy cannot
consume its shutdown message while blocked accepting the unrelated connection's
incomplete NCCL handshake. Its origin (for example a port scanner) is unknown.

NCCL's normal interface selection can choose the server's public interface.
`MASTER_ADDR=127.0.0.1` only configures the PyTorch rendezvous store; it does not
restrict NCCL's separate bootstrap/proxy listeners. The existing launch scripts
therefore did not prevent this external connection, even for one GPU/rank.

The six DataLoader workers did inherit other socket descriptors, as confirmed by
`ss`, but did not inherit fd 82. Their presence alone is not evidence that they
caused this hang. Train/validation inherit `persistent_workers=True`, with four
and two workers respectively; these intentionally survive iterator exhaustion.
The original cleanup kept them alive until after process-group destruction.

References:

* [NCCL 2.26.2 socket handshake implementation](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/misc/socket.cc)
* [NCCL proxy service and shutdown](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/proxy.cc)
* [NCCL incomplete external handshake issue and proposed native fix](https://github.com/NVIDIA/nccl/pull/1880)
* [NVIDIA interface-selection documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)

## Changes

Before Runner construction, Linux torchrun jobs with `WORLD_SIZE=1` explicitly
select `NCCL_SOCKET_IFNAME==lo` (environment value `=lo`, exact interface match).
A single rank has no remote peers, so its NCCL sockets should be local. This
removes the public listener through which the observed external connection
entered. It also applies when invoking train/test directly with torchrun.
An inherited interface selection is deliberately overridden for this case.
Already initialized groups, other platforms and jobs with more than one rank
retain their existing interface settings. This configures endpoint binding; it
does not patch the installed NCCL library or claim to repair its handshake
implementation for multi-rank jobs.

After Runner returns, stop already-built DataLoader iterators using PyTorch's
own worker shutdown routine, then destroy all process groups. Include MMEngine's
iteration-based iterator and deduplicate shared iterators. Never access lazy
Runner loop properties or create new iterators during cleanup. Continue other
cleanup on worker errors and preserve the original training/testing exception.
Worker cleanup is a separate resource-lifetime correction, not the explanation
for fd 82's handshake wait.

No changes to the model, support memory, data pipelines, random seeds,
persistent-worker settings, validation schedule, optimizer, checkpoint hooks,
checkpoint selection, prediction dumps, or train-then-test shell sequence.
All final validation/checkpoint/after-run work remains inside `runner.train()`
or `runner.test()`, before cleanup starts. No forced process exit, skipped
NCCL cleanup, additional barrier, or production shutdown timer is used.

## Validation

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

On the Windows development host, 53 tests ran: 46 passed and seven were
skipped because MMCV/MMEngine/pycocotools were unavailable. `git diff --check`
also passed.

The tests include real PyTorch 2.7.1 CPU/Gloo subprocess lifecycles with four
train and two validation persistent workers, auxiliary groups, a saved tensor
checkpoint, subsequent checkpoint loading in a separate Test process with
non-persistent workers, and failure cleanup. These are lifecycle smoke tests;
they do not run the detector or MMEngine checkpoint hooks. The existing model
regressions cover losses, gradients, support memory, configuration and logging.

On the Linux training server, the same smoke can exercise CUDA/NCCL and assert
the actual native listeners use loopback before auxiliary Gloo groups are built:

```bash
CUDA_VISIBLE_DEVICES=0 CDFSOD_SHUTDOWN_BACKEND=nccl \
  python -m unittest discover -s tests -p test_distributed_runtime_integration.py -v
```

This developer smoke uses a parent timeout only to bound the test process. It
does not change the production runtime. CUDA/NCCL and a full NEU-DET rerun were
not available on the Windows development host. Validate those on the server.
The code change affects newly started processes; the already stuck process
still has its original public-interface NCCL communicator.
