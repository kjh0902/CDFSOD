"""CPU-only lifetime regressions; no CUDA, MMCV or downloaded weights needed."""
import ast
import importlib.util
import io
import os
import sys
import unittest
from visual_test_support import compat
from contextlib import redirect_stderr
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DistributedRuntimeTests(unittest.TestCase):
    def setUp(self):
        # Load the actual helper without importing the heavy mmdet package.
        self.dist = ModuleType('torch.distributed')
        self.dist.is_available = Mock(return_value=True)
        self.dist.is_initialized = Mock(return_value=True)
        self.dist.destroy_process_group = Mock()
        torch = ModuleType('torch')
        torch.distributed = self.dist
        spec = importlib.util.spec_from_file_location(
            'distributed_runtime',
            ROOT / 'mmdet/utils/distributed_runtime.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch,
                                    'torch.distributed': self.dist}):
            spec.loader.exec_module(module)
        self.configure_nccl = module._configure_single_rank_nccl
        self.runtime = module.distributed_runtime
        self.shutdown_workers = module._shutdown_runner_workers
        self.stderr = io.StringIO()
        self.capture = redirect_stderr(self.stderr)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def test_cleanup_only_after_body_and_for_all_groups(self):
        with self.runtime():
            self.dist.destroy_process_group.assert_not_called()
        self.dist.destroy_process_group.assert_called_once_with()
        self.assertIn('process groups destroyed', self.stderr.getvalue())

    def test_non_distributed_and_unavailable_are_noops(self):
        self.dist.is_initialized.return_value = False
        with self.runtime():
            pass
        self.dist.is_available.return_value = False
        self.dist.is_initialized.reset_mock()
        with self.runtime():
            pass
        self.dist.is_initialized.assert_not_called()
        self.dist.destroy_process_group.assert_not_called()

    def test_initialization_is_checked_at_exit(self):
        self.dist.is_initialized.return_value = False
        with self.runtime():
            self.dist.is_initialized.return_value = True
        self.dist.destroy_process_group.assert_called_once_with()

    def test_runtime_exception_and_interrupt_propagate(self):
        for error in (RuntimeError('training failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.dist.destroy_process_group.reset_mock()
                with self.assertRaises(type(error)) as caught:
                    with self.runtime():
                        raise error
                self.assertIs(caught.exception, error)
                self.dist.destroy_process_group.assert_called_once_with()

    def test_cleanup_failure_is_not_reported_as_success(self):
        self.dist.destroy_process_group.side_effect = RuntimeError('cleanup')
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            with self.runtime():
                pass
        self.assertNotIn('process groups destroyed', self.stderr.getvalue())

    def test_cleanup_failure_does_not_mask_original_exception(self):
        self.dist.destroy_process_group.side_effect = RuntimeError('cleanup')
        original = ValueError('original')
        with self.assertRaises(ValueError) as caught:
            with self.runtime():
                raise original
        self.assertIs(caught.exception, original)
        self.assertIn('RuntimeError: cleanup', self.stderr.getvalue())

    def run_entrypoint(self, mode, custom=False, failure=None):
        # Execute the unchanged main function AST, replacing only dependencies.
        tree = ast.parse((ROOT / f'tools/{mode}.py').read_text('utf-8'))
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        cfg = Config(work_dir='existing-work-dir')
        if custom:
            cfg.runner_type = 'CustomRunner'
        args = SimpleNamespace(
            config='original.py', launcher='pytorch', cfg_options=None,
            work_dir=None, amp=False, auto_scale_lr=False, resume=None,
            checkpoint='best.pth', show=False, show_dir=None, tta=False,
            out='results.pkl')
        events = []
        runner = SimpleNamespace(
            logger=Mock(), test_evaluator=SimpleNamespace(metrics=[]),
            _train_loop=None, _val_loop=None, _test_loop=None)
        shutdown = Mock(side_effect=lambda: events.append('workers'))
        loop = SimpleNamespace(dataloader=SimpleNamespace(
            _iterator=SimpleNamespace(_shutdown_workers=shutdown)))
        setattr(runner, f'_{mode}_loop', loop)

        def run():
            self.assertEqual(cfg.launcher, 'pytorch')
            self.dist.destroy_process_group.assert_not_called()
            if mode == 'test':
                self.assertEqual(cfg.load_from, 'best.pth')
                self.assertEqual(runner.test_evaluator.metrics, ['results.pkl'])
            events.append('run')
            if failure == 'run':
                raise RuntimeError('run failed')
            events.append('after_run')

        setattr(runner, mode, run)

        def build(config):
            self.assertEqual(config, cfg)
            self.dist.destroy_process_group.assert_not_called()
            events.append('build')
            if failure == 'build':
                raise RuntimeError('build failed')
            return runner

        builder = Mock(side_effect=build)
        self.dist.destroy_process_group.reset_mock()
        self.dist.destroy_process_group.side_effect = lambda: events.append('destroy')
        env = dict(
            parse_args=lambda: args,
            setup_cache_size_limit_of_dynamo=lambda: None,
            Config=SimpleNamespace(fromfile=lambda path: cfg),
            Runner=SimpleNamespace(from_cfg=builder),
            RUNNERS=SimpleNamespace(build=builder, get=lambda name: SimpleNamespace()),
            runner_randomness_config=compat.runner_randomness_config,
            apply_warn_only=compat.apply_warn_only,
            DumpDetResults=lambda out_file_path: out_file_path,
            distributed_runtime=self.runtime)
        exec(compile(ast.Module(body=[main], type_ignores=[]),
                     f'tools/{mode}.py', 'exec'), env)
        if failure:
            with self.assertRaisesRegex(RuntimeError, f'{failure} failed'):
                env['main']()
            runner.logger.info.assert_not_called()
        else:
            env['main']()
            runner.logger.info.assert_called_once()
        self.dist.destroy_process_group.assert_called_once_with()
        expected = ['build']
        if failure != 'build':
            expected.append('run')
        if failure is None:
            expected.append('after_run')
        if failure != 'build':
            expected.append('workers')
        self.assertEqual(events, expected + ['destroy'])

    def test_train_and_test_normal_and_custom_runner_lifetimes(self):
        for mode in ('train', 'test'):
            for custom in (False, True):
                with self.subTest(mode=mode, custom=custom):
                    self.run_entrypoint(mode, custom)

    def test_train_and_test_construction_and_execution_failures(self):
        for mode in ('train', 'test'):
            for custom in (False, True):
                for failure in ('build', 'run'):
                    with self.subTest(mode=mode, custom=custom, failure=failure):
                        self.run_entrypoint(mode, custom, failure)

    def test_workers_stop_before_groups_on_every_exit_path(self):
        for error in (None, ValueError('training'), KeyboardInterrupt()):
            with self.subTest(error=error):
                events = []
                shutdown = Mock(side_effect=lambda: events.append('workers'))
                runner = SimpleNamespace(_train_loop=SimpleNamespace(
                    dataloader=SimpleNamespace(_iterator=SimpleNamespace(
                        _shutdown_workers=shutdown))))
                self.dist.destroy_process_group.side_effect = lambda: events.append('groups')

                def run():
                    with self.runtime(lambda: runner):
                        events.extend(['train', 'validation', 'checkpoint', 'after_run'])
                        shutdown.assert_not_called()
                        if error:
                            raise error

                if error:
                    with self.assertRaises(type(error)) as caught:
                        run()
                    self.assertIs(caught.exception, error)
                else:
                    run()
                self.assertEqual(events, ['train', 'validation', 'checkpoint',
                                          'after_run', 'workers', 'groups'])

    def test_does_not_build_unused_loops_or_create_iterators(self):
        class LazyRunner:
            _train_loop = dict(type='UnbuiltLoop')
            _val_loop = None
            _test_loop = SimpleNamespace(dataloader=SimpleNamespace(_iterator=None))

            @property
            def train_dataloader(self):
                raise AssertionError('cleanup must not build a DataLoader')

            val_dataloader = test_dataloader = train_dataloader
            train_loop = val_loop = test_loop = train_dataloader

        with self.runtime(lambda: LazyRunner()):
            pass
        self.dist.destroy_process_group.assert_called_once_with()

    def test_shared_and_infinite_iterators_and_single_process_loader(self):
        shutdown = Mock()
        iterator = SimpleNamespace(_shutdown_workers=shutdown)
        loop = SimpleNamespace(
            dataloader=SimpleNamespace(_iterator=iterator),
            dataloader_iterator=SimpleNamespace(_iterator=iterator))
        runner = SimpleNamespace(_train_loop=loop, _val_loop=loop,
                                 _test_loop=SimpleNamespace(dataloader=SimpleNamespace(
                                     _iterator=object())))
        self.shutdown_workers(runner)
        shutdown.assert_called_once_with()
        # IterBasedTrainLoop can own the only reference to a worker iterator.
        shutdown.reset_mock()
        loop.dataloader._iterator = None
        self.shutdown_workers(runner)
        shutdown.assert_called_once_with()

    def test_workers_close_without_distributed_initialization(self):
        self.dist.is_initialized.return_value = False
        shutdown = Mock()
        runner = SimpleNamespace(_val_loop=SimpleNamespace(
            dataloader=SimpleNamespace(_iterator=SimpleNamespace(
                _shutdown_workers=shutdown))))
        with self.runtime(lambda: runner):
            shutdown.assert_not_called()
        shutdown.assert_called_once_with()
        self.dist.destroy_process_group.assert_not_called()

    def test_worker_failure_still_closes_other_loaders_and_groups(self):
        error = RuntimeError('worker shutdown')
        broken = SimpleNamespace(_shutdown_workers=Mock(side_effect=error))
        healthy = SimpleNamespace(_shutdown_workers=Mock())
        runner = SimpleNamespace(
            _train_loop=SimpleNamespace(dataloader=SimpleNamespace(_iterator=broken)),
            _val_loop=SimpleNamespace(dataloader=SimpleNamespace(_iterator=healthy)))
        with self.assertRaises(RuntimeError) as caught:
            with self.runtime(lambda: runner):
                pass
        self.assertIs(caught.exception, error)
        healthy._shutdown_workers.assert_called_once_with()
        self.dist.destroy_process_group.assert_called_once_with()
        self.dist.destroy_process_group.reset_mock()
        self.dist.destroy_process_group.side_effect = RuntimeError('group shutdown')
        original = ValueError('original training failure')
        with self.assertRaises(ValueError) as caught:
            with self.runtime(lambda: runner):
                raise original
        self.assertIs(caught.exception, original)
        self.dist.destroy_process_group.assert_called_once_with()



    def test_single_rank_nccl_uses_loopback_before_runner_initialization(self):
        self.dist.is_initialized.return_value = False
        for old_interface in (None, 'eth0'):
            with self.subTest(interface=old_interface):
                env = {'WORLD_SIZE': '1', 'MASTER_ADDR': '127.0.0.1'}
                if old_interface:
                    env['NCCL_SOCKET_IFNAME'] = old_interface
                with patch.dict(os.environ, env, clear=True), patch.object(sys, 'platform', 'linux'):
                    with self.runtime():
                        self.assertEqual(os.environ['NCCL_SOCKET_IFNAME'], '=lo')
                        self.dist.is_initialized.return_value = True
                self.dist.is_initialized.return_value = False

    def test_multi_rank_and_non_torchrun_network_settings_are_preserved(self):
        self.dist.is_initialized.return_value = False
        for world_size in (None, '2', '8'):
            for platform in ('linux', 'win32'):
                with self.subTest(world_size=world_size, platform=platform):
                    env = {'NCCL_SOCKET_IFNAME': '=ib0'}
                    if world_size:
                        env['WORLD_SIZE'] = world_size
                    with patch.dict(os.environ, env, clear=True), patch.object(sys, 'platform', platform):
                        self.configure_nccl()
                        self.assertEqual(os.environ['NCCL_SOCKET_IFNAME'], '=ib0')
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}, clear=True):
            with patch.object(sys, 'platform', 'win32'):
                self.configure_nccl()
                self.assertNotIn('NCCL_SOCKET_IFNAME', os.environ)

    def test_does_not_reconfigure_existing_groups_or_unavailable_backend(self):
        for available, initialized in ((True, True), (False, False)):
            with self.subTest(available=available, initialized=initialized):
                self.dist.is_available.return_value = available
                self.dist.is_initialized.return_value = initialized
                with patch.dict(os.environ, {'WORLD_SIZE': '1'}, clear=True), patch.object(sys, 'platform', 'linux'):
                    self.configure_nccl()
                    self.assertNotIn('NCCL_SOCKET_IFNAME', os.environ)


if __name__ == '__main__':
    unittest.main()
