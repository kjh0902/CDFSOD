# UODD final-validation SIGABRT investigation

Baseline: `1ae964e` on `codex/acl-raw-mean-prototype-etf-loss`.
Reported environment: Python 3.10.20, torch 2.7.1+cu128,
torchvision 0.22.1+cu128, MMEngine 0.10.7, RTX 5090, one process launched
with `--launcher pytorch`.

## Evidence and limits

The supplied log shows checkpoint saving and COCO evaluation finishing, followed
by `munmap_chunk(): invalid pointer` and SIGABRT. This is a native allocator
failure, not a Python exception. The elastic `ChildFailedError` reports the
worker's death; it does not identify the component that corrupted or freed the
memory. Corruption can also be detected later than the operation that caused it.
Successful NEU-DET runs do not isolate a particular UODD operator as the cause.

There is a concrete lifecycle omission: both CLI entrypoints initialized
distributed execution through MMEngine but never called
`torch.distributed.destroy_process_group()`. A world size of one still creates
the process group. MMEngine 0.10.7's `Runner.train()`/`test()` execute
`after_run` hooks and return, without destroying process groups.
The final metric log occurs before all remaining hooks and runner teardown have
necessarily finished.

[PyTorch 2.7's shutdown guidance](https://docs.pytorch.org/docs/2.7/distributed.html#shutdown)
recommends explicit process-group destruction before interpreter shutdown; it
avoids leaving NCCL abort to nondeterministic destructor ordering.
[MMEngine's runner source](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/runner/runner.py)
confirms the return path above. This establishes a missing cleanup operation,
**not proof that NCCL caused this particular allocator abort**. A native
backtrace or reproduction on the affected server is still needed to establish
that causal link. No RTX 5090/UODD reproduction was available in the local
Windows editing environment.

## Change

Both entrypoints now enclose runner construction and execution in a shared
`try/finally` context. After the runner returns, it destroys all initialized
process groups while Python is still running, including auxiliary groups.
Cleanup also runs on Python exceptions and interrupts, without adding a barrier
that could wait for a failed peer. Cleanup failures remain failures; if training
already failed, its original exception is preserved and the cleanup error is
printed as well. Native SIGABRT cannot be caught by this context.

The entrypoints enable `faulthandler` and print completion/cleanup markers so a
recurrence can be localized. They do not suppress signals, force a successful
exit, or skip validation. Model code, raw-mean ETF weight (0.1), seeds, optimizer,
scheduler, transforms, batch sizes, workers, persistent workers, multiprocessing
start method, precision, launcher and dependency versions are unchanged.

## Verification and server rerun

The CPU-only regression suite covers normal and custom runners, construction and
execution failures, interrupts, no process group, and cleanup failure handling:

```bash
python -B -m unittest discover -s tests -p test_distributed_runtime.py -v
```

These tests use a stub distributed backend and do not validate NCCL or claim to
reproduce the native crash. For server verification, run the full original
training schedule into a **fresh** work directory (preserve the existing results):

```bash
set -o pipefail
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTHONFAULTHANDLER=1
ulimit -c unlimited
bash tools/dist_train.sh \
  configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_UODD_1shot.py \
  1 29500 0 --work-dir exp_cdfsod_shutdown_check/UODD/1shot \
  2>&1 | tee uodd_shutdown_check.log
```

The NCCL environment variables above match `run_cdfsod.sh`. Do not use a completed
checkpoint with `--resume` as a substitute for reproducing the full final
training/validation path. Check the process exit code as well as the metrics.

Expected tail on a successful distributed training run:

```text
Runner.train() completed; starting shutdown.
[shutdown] Destroying distributed process groups.
[shutdown] Distributed process groups destroyed.
```

If the failure recurs, preserve the entire log and core dump. A failure before
the runner completion marker can still be in a hook or an earlier native
operation; between cleanup markers it is inside process-group destruction;
after both markers it is in later teardown. These are locations of detection,
not necessarily locations of corruption. `faulthandler` on Python 3.10 supplies
Python thread stacks, not a full C/C++ backtrace. Use the server's core-dump
facility and GDB (`thread apply all bt`) to identify the native library before
changing data-loader settings, numerical precision or package versions.
