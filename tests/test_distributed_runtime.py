"""CPU-only lifetime regressions; no CUDA, MMCV or downloaded weights needed."""
import ast
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    def __getattr__(self, name):
        return self[name]

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
        self.runtime = module.distributed_runtime
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
            logger=Mock(), test_evaluator=SimpleNamespace(metrics=[]))

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
            self.assertIs(config, cfg)
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
            RUNNERS=SimpleNamespace(build=builder),
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


if __name__ == '__main__':
    unittest.main()
