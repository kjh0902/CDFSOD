"""Exercise weight/GPU/work-dir propagation through real Bash launchers."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
if os.name == 'nt' and Path('C:/Program Files/Git/bin/bash.exe').is_file():
    BASH = 'C:/Program Files/Git/bin/bash.exe'
OPTIONS = ['bert-etf-loss-weight', 'bert-orth-loss-weight',
           'fe-etf-loss-weight', 'fe-orth-loss-weight']


@unittest.skipUnless(BASH, 'Requires Bash')
class RawMeanCLIIntegrationTests(unittest.TestCase):
    def run_cli(self, *args, root=ROOT, env=None):
        return subprocess.run([BASH, (root / 'run_cdfsod.sh').as_posix(), *args],
                              cwd=root, env=env, capture_output=True,
                              text=True, encoding='utf-8', timeout=20)

    def test_all_dataset_shot_defaults_and_gpu_choices(self):
        for dataset in ['ArTaxOr', 'Clipart1k', 'DIOR', 'FISH', 'NEU-DET', 'UODD']:
            for shot in ['1', '5', '10']:
                for gpu in ['0', '1']:
                    result = self.run_cli('--dataset', dataset, '--shot', shot,
                                          '--gpu', gpu, '--dry-run')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f'physical {gpu} -> cuda:0', result.stdout)
                    for option in OPTIONS:
                        self.assertIn(f'{option.replace("-", "_")}: from config', result.stdout)
        first = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', '--dry-run')
        second = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', '--dry-run')
        self.assertNotEqual(next(l for l in first.stdout.splitlines() if l.startswith('Output')),
                            next(l for l in second.stdout.splitlines() if l.startswith('Output')))

    def test_invalid_arguments_and_weights(self):
        for option in OPTIONS + ['etf-loss-weight']:
            for value in ['', '-0.1', 'nan', 'inf', '1e999', 'bad', '1; exit 0']:
                result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1',
                                      f'--{option}', value, '--dry-run')
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', f'--{option}')
            self.assertEqual(result.returncode, 2)
        for option, value in [('gpu', '2'), ('gpu', '0,1'), ('gpu', '-1'),
                              ('shot', '3'), ('dataset', 'unknown'), ('port', '65536')]:
            result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1',
                                  f'--{option}', value, '--dry-run')
            self.assertEqual(result.returncode, 2)
        for extra in [('--resume',), ('--work-dir', '')]:
            result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', *extra)
            self.assertEqual(result.returncode, 2)

    def test_weights_resume_and_single_physical_gpu_reach_train_and_test(self):
        with tempfile.TemporaryDirectory(prefix='acl-hed-mu-cli-') as directory:
            root = Path(directory)
            (root / 'run_cdfsod.sh').write_text((ROOT / 'run_cdfsod.sh').read_text(encoding='utf-8'),
                                               encoding='utf-8', newline='\n')
            configs = root / 'configs_cdfsod/final_configs_bs4'
            configs.mkdir(parents=True)
            (configs / 'grounding_dino_swin-b_finetune_NEU-DET_1shot.py').touch()
            tools = root / 'tools'
            tools.mkdir()
            for name in ['dist_train.sh', 'dist_test.sh']:
                (tools / name).write_text((ROOT / 'tools' / name).read_text(encoding='utf-8'),
                                          encoding='utf-8', newline='\n')
            binary = root / 'bin'
            binary.mkdir()
            python = binary / 'python'
            python.write_text('''#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *train.py*) kind=train ;;
  *test.py*) kind=test ;;
  *) exit 1 ;;
esac
printf '%s\\n' "$@" > "${CAPTURE_ROOT}/${kind}_args.txt"
printf '%s\\n' "${CUDA_VISIBLE_DEVICES}" "${CUDA_DEVICE_ORDER}" "${NNODES}" "${NODE_RANK}" > "${CAPTURE_ROOT}/${kind}_env.txt"
if [[ "$kind" == train ]]; then
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --work-dir ]]; then
      touch "$2/best_coco_bbox_mAP_iter_1.pth"
      break
    fi
    shift
  done
fi
''', encoding='utf-8', newline='\n')
            python.chmod(0o755)
            bash_path = subprocess.check_output([BASH, '-c', 'printf "%s" "$PATH"'],
                                                text=True, encoding='utf-8').strip()
            env = dict(os.environ, PATH=binary.as_posix() + ':' + bash_path,
                       CAPTURE_ROOT=root.as_posix(), NNODES='2', NODE_RANK='1')
            for gpu in ['0', '1']:
                output = (root / f'experiment {gpu}').as_posix()
                result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1', '--gpu', gpu,
                                      '--work-dir', output, '--resume',
                                      '--bert-etf-loss-weight', '0', '--bert-orth-loss-weight', '.3',
                                      '--fe-etf-loss-weight', '1e-2', '--fe-orth-loss-weight', '2',
                                      root=root, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for kind in ['train', 'test']:
                    args = (root / f'{kind}_args.txt').read_text(encoding='utf-8').splitlines()
                    settings = (root / f'{kind}_env.txt').read_text(encoding='utf-8').splitlines()
                    self.assertEqual(settings, [gpu, 'PCI_BUS_ID', '1', '0'])
                    self.assertIn('--nproc_per_node=1', args)
                    self.assertEqual(args[args.index('--work-dir') + 1], output)
                    for name, weight in zip(OPTIONS, ['0', '.3', '1e-2', '2']):
                        self.assertIn(f'model.{name.replace("-", "_")}={weight}', args)
                    self.assertEqual('--resume' in args, kind == 'train')
                for alias in ['--etf-loss-weight', '--raw-mean-etf-loss-weight']:
                    result = self.run_cli('--dataset', 'NEU-DET', '--shot', '1',
                                          '--work-dir', output, alias, '.25', root=root, env=env)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('model.bert_etf_loss_weight=.25',
                                  (root / 'train_args.txt').read_text(encoding='utf-8').splitlines())


if __name__ == '__main__':
    unittest.main()
